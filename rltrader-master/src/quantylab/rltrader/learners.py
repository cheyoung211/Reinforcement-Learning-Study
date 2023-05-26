import os
import logging
import abc
import collections
import threading
import time
import json
import numpy as np
from tqdm import tqdm
from quantylab.rltrader.environment import Environment
from quantylab.rltrader.agent import Agent
from quantylab.rltrader.networks import Network, DNN, LSTMNetwork, CNN
from quantylab.rltrader.visualizer import Visualizer
from quantylab.rltrader import utils
from quantylab.rltrader import settings


logger = logging.getLogger(settings.LOGGER_NAME)


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None, 
                chart_data=None, training_data=None,
                min_trading_price=100000, max_trading_price=10000000, 
                net='dnn', num_steps=1, lr=0.0005, 
                discount_factor=0.9, num_epoches=1000,
                balance=100000000, start_epsilon=1,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True, gen_output=True):
        # 인자 확인
        # 투자 최소 및 최대 단위
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method  # 강화학습 기법
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 환경 설정
        self.stock_code = stock_code  # 주식 종목 코드
        self.chart_data = chart_data  # 일봉 차트 데이터
        self.environment = Environment(chart_data)  # chart_data를 인자로 하여 Environment 클래스의 인스턴스를 생성
        # 에이전트 설정
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price)  # 강화학습 환경을 인자로 Agent 클래스의 인스턴스 생성
        # 학습 데이터
        self.training_data = training_data  # 전처리된 학습 데이
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net  # 신경망 종류
        self.num_steps = num_steps  # lstm, cnn에서 사용하는 step
        self.lr = lr
        self.value_network = value_network  # 가치 신경망
        self.policy_network = policy_network  # 정책 신경망
        self.reuse_models = reuse_models  # reuse_models가 True이고 모델이 이미 존재하는 경우 재활
        # 가시화 모듈
        self.visualizer = Visualizer()  # 에포크마다 투자 결과를 가시화 하기 위해 Visualizer 클래스 객체 생성
        # 메모리 - 강화학습 과정에서 발생하는 각종 데이터를 저장하기 위한 변수
        self.memory_sample = []  # 학습 데이터 샘플
        self.memory_action = []  # 수행한 행동
        self.memory_reward = []  # 획득한 보상
        self.memory_value = []  # 행동의 예측 가치
        self.memory_policy = []  # 행동의 예측 확률
        self.memory_pv = []  # 포트폴리오 가치
        self.memory_num_stocks = []  #보유 주식 수
        self.memory_exp_idx = []  # 탐험 위치
        # 에포크 관련 정보
        self.loss = 0.  # 에포크 동안 학습에서 발생한 손실
        self.itr_cnt = 0  # 수익 발생 횟수
        self.exploration_cnt = 0  # 탐험 횟수
        self.batch_size = 0  # 학습 횟
        # 로그 등 출력 경로
        self.output_path = output_path  # 학습 과정에서 발생하는 로그, 가시화결과 및 학습 종료 후 저장되는 신경망 모델 파일은 지정된 경로에 저장
        self.gen_output = gen_output

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):  # 지정된 신경망 종류에 맞게 가치 신경망 생성 - 가치 신경망 : 손익률 회귀 분석 모델
        if self.net == 'dnn':  
            self.value_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network,  # shared_network : 신경망의 상단부 - 여러 신경망이 공유할 수 있음
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.value_network_path):  # reuse_models가 True이고, value_network_path에 지정된 경로에 신경망 모델 파일이 존재하면
            self.value_network.load_model(model_path=self.value_network_path)  #신경망 모델 파일 불러오기

    def init_policy_network(self, shared_network=None, activation='sigmoid', 
                            loss='binary_crossentropy'):  # 정책 신경망 함수와 매우 유사, activation 으로 sigmoid 함수 사용
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self):
        pass

    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            self.loss = loss

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS, 
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(self.epoch_summary_dir, f'epoch_summary_{epoch_str}.png'))

    def run(self, learning=True):
        info = (
            f'[{self.stock_code}] RL:{self.rl_method} NET:{self.net} '
            f'LR:{self.lr} DF:{self.discount_factor} '
        )
        with self.lock:
            logger.debug(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        if self.gen_output:
            self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
            if not os.path.isdir(self.epoch_summary_dir):
                os.makedirs(self.epoch_summary_dir)
            else:
                for f in os.listdir(self.epoch_summary_dir):
                    os.remove(os.path.join(self.epoch_summary_dir, f))

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 에포크 반복
        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = self.start_epsilon * (1 - (epoch / (self.num_epoches - 1)))
            else:
                epsilon = self.start_epsilon

            for i in tqdm(range(len(self.training_data)), leave=False):
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = \
                    self.agent.decide_action(pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 보상 획득
                reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

            # 에포크 종료 후 학습
            if learning:
                self.fit()

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(self.num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            logger.debug(f'[{self.stock_code}][Epoch {epoch_str}/{self.num_epoches}] '
                f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')

            # 에포크 관련 정보 가시화
            if self.gen_output:
                if self.num_epoches == 1 or (epoch + 1) % max(int(self.num_epoches / 10), 1) == 0:
                    self.visualize(epoch_str, self.num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logger.debug(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f} '
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

    def predict(self):
        # 에이전트 초기화
        self.agent.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        
        result = []
        while True:
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # num_steps만큼 샘플 저장
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # 가치, 정책 신경망 예측
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample)).tolist()
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample)).tolist()
            
            # 신경망에 의한 행동 결정
            result.append((self.environment.observation[0], pred_value, pred_policy))

        if self.gen_output:
            with open(os.path.join(self.output_path, f'pred_{self.stock_code}.json'), 'w') as f:
                print(json.dumps(result), file=f)

        return result


class DQNLearner(ReinforcementLearner):  # DQN 강화학습 클래스 - 가치 신경망으로만 학습
    def __init__(self, *args, value_network_path=None, **kwargs):  # 생성자
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()  # 가치 신경망 - line 86

    def get_batch(self):  # 배치학습 데이터 생성 
        memory = zip(  # 메모리 배열을 역으로 해서 묶어줌 - 마지막 데이터가 가장 최근 데이터이기 때문
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))  # 샘플 배열(모두 0으로)
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))  # lable 배열(모두 0으로)
        value_max_next = 0
        for i, (sample, action, value, reward) in enumerate(memory):  # for 문 돌면서 샘플 배열과 lable 배열에 값 채우기
            x[i] = sample
            r = self.memory_reward[-1] - reward  # r : 학습에 사용할 보상, reward : 행동을 수행한 시점에서의 손익률
            y_value[i] = value
            y_value[i, action] = r + self.discount_factor * value_max_next  # 보상 + 할인률 * 다음 상태의 최대 가치
            value_max_next = value.max()
        return x, y_value, None  # 샘플 배열, 가치 신경망 학습 레이블 배열, 정책 신경망 학습 레이블 배열(DQN은 정책 신경망 사용x -> None)


class PolicyGradientLearner(ReinforcementLearner):  # 정책 경사 강화학습 클래스 - 정책 신경망으로만 학습
    def __init__(self, *args, policy_network_path=None, **kwargs):  # DQN과 동일
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path 
        self.init_policy_network()  # 정책 신경망 - line 110

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))  # 학습 데이터 및 에이전트 상태로 구성된 샘플 배열 - 배치 데이터 크기, 학습 데이터 특징 크 
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))  # 정책 신경망 학습을 위한 레이블 배열 - 배치 데이터 크기, 에이전트 행동의 수
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(r)  # sigmoid 사용
        return x, None, y_policy  # 정책 경사는 가치 신경망이 없으므로 두 번째 반환값 None


class ActorCriticLearner(ReinforcementLearner):  # Actor-Critic 강화학습 클래스 - 가치 신경망과 정책 신경망 모두 사용
    def __init__(self, *args, shared_network=None, 
        value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:  # 가치 신경망과 정책 신경망의 상단부 레이어 공유
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps, 
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        # y_value와 y_policy를 모두 가짐
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i, :] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(r)
            value_max_next = value.max()
        return x, y_value, y_policy


class A2CLearner(ActorCriticLearner):  # 액터-크리틱을 상
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_batch(self):  # advantage(상태-행동 가치에 상태 가치를 뺀 값)로 정책 신경망 학습
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = reward_next + self.memory_reward[-1] - reward * 2
            reward_next = reward
            y_value[i, :] = value
            y_value[i, action] = np.tanh(r + self.discount_factor * value_max_next)
            advantage = y_value[i, action] - y_value[i].mean()  # 가치 신경망의 예측 상태-행동 가치 값의 균평균
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(advantage)
            value_max_next = value.max()
        return x, y_value, y_policy


class A3CLearner(ReinforcementLearner):  # A3C 강화학습 클래스 (A2C를 병렬적으로 수행)
    def __init__(self, *args, list_stock_code=None,   # A2C와 달리 리스트로 학습할 종목에 대한 여러 데이터를 인자로 받음
        list_chart_data=None, list_training_data=None,
        list_min_trading_price=None, list_max_trading_price=None, 
        value_network_path=None, policy_network_path=None,
        **kwargs):
        assert len(list_training_data) > 0  # 리스트의 크키만큼 A2C 클래스의 객체 생성
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(
            net=self.net, num_steps=self.num_steps, 
            input_dim=self.num_features,
            output_dim=self.agent.NUM_ACTIONS)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data, 
            min_trading_price, max_trading_price) in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_price, list_max_trading_price
            ):
            learner = A2CLearner(*args, 
                stock_code=stock_code, chart_data=chart_data, 
                training_data=training_data,
                min_trading_price=min_trading_price, 
                max_trading_price=max_trading_price, 
                shared_network=self.shared_network,
                value_network=self.value_network,
                policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)

    def run(self, learning=True):  # A3C 수행함수 - A2C를 병렬로 동시 수행
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.run, daemon=True, kwargs={'learning': learning}  # A2C 클래스 객체의 run 함수 동시 수행
            ))
        # 모든 A2C 강화학습 이후에 A3C
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def predict(self):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.predict, daemon=True
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
