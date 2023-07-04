import os
import sys
import logging
import argparse
import json

from quantylab.rltrader import settings
from quantylab.rltrader import utils
from quantylab.rltrader import data_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4', 'v4.1', 'v4.2'], default='v4.1')  #RLTrader의 버전
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--stock_code', nargs='+')  #주식 종목 코드 (A3C의 경우 여러개 입력)
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'], default='a2c')  #강화학습 방식
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')  #가치 신경망과 정책 신경망에서 사용할 신경망 선택
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')  #keras의 백엔드로 사용할 프레임워크
    parser.add_argument('--start_date', default='20200101')  #차트 데이터 및 학습 데이터 시작 날짜
    parser.add_argument('--end_date', default='20201231')  #차트 데이터 및 학습 데이터 끝 날짜
    parser.add_argument('--lr', type=float, default=0.001)  #학습 속도
    parser.add_argument('--discount_factor', type=float, default=0.7)  #할인율
    parser.add_argument('--balance', type=int, default=100000000)  #초기 자본금
    args = parser.parse_args()

    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'  #로그, 가시화 파일, 신경망 모델 등의 출력 파일을 저장할 폴더의 이름
    learning = args.mode in ['train', 'update']  #강화학습 유무
    reuse_models = args.mode in ['test', 'update', 'predict']  #신경망 모델 재사용 유무
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'  #가치 신경망 모델 파일명
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl' #정책 신경망 모델 파일명
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0  #시작 탐험률
    num_epoches = 100 if args.mode in ['train', 'update'] else 1  #epoch 수
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1  #lstm과 cnn에서 사용할 step 크기

    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):  #만약 directory가 존재하지 않으면 만들기
        os.makedirs(output_path)

    # 파라미터 기록
    params = json.dumps(vars(args))  #입력받은 인자를 json 형식으로 저장
    with open(os.path.join(output_path, 'params.json'), 'w') as f:  #경로에 dictoinary로 만들어서 'params.json' 파일에 json 형태로 저장, 출력 경로에 로그 파일 성생성
        f.write(params)

    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    # 'models'폴더에 파일 준비 - 모델 재사용시 편하게 모델을 지정할 수 있음
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)  
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from quantylab.rltrader.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner
    
    #강화학습 실행
    common_params = {}  #공통 parameter - 강화학습 방법, 지연보상 임곗값, 신경망 종류, steop수, 출력 경로, 신경망 모델 재사용 여부 
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver)

        assert len(chart_data) >= num_steps
        
        # 최소/최대 단일 매매 금액 설정
        min_trading_price = 100000
        max_trading_price = 10000000

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
            'balance': args.balance, 'num_epoches': num_epoches, 
            'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
            'output_path': output_path, 'reuse_models': reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':  # 강화학습 종류에 맞춰 학습기 클래스를 정하고 가치 신경망과 정책 신경망의 경로 
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_price': min_trading_price, 
                'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)

    if args.rl_method == 'a3c':  #A3C 강화학습을 위해 A3CLearner 클래스 객체를 생성
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price, 
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
    
    assert learner is not None

    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)  #강화학습 시작
        if args.mode in ['train', 'update']:
            learner.save_models()  #학습한 신경망 모델 저장
    elif args.mode == 'predict':
        learner.predict()
