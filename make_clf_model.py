#-*- coding: utf-8 -*-
import pandas as pd
import os
import subprocess
import logging
import requests
import joblib #import sklearn.external.joblib as extjoblib
import set_clf_models

from datetime import datetime, timezone, timedelta
from elasticsearch import Elasticsearch, helpers
from configparser import ConfigParser
from logging.handlers import RotatingFileHandler
from multiprocessing import Process

# #모델 평가
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

# #시각화
import matplotlib.pyplot as plt
import seaborn as sns
#
# #데이터
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN

class MAKE_CLF_MODELS(Process):
    def __init__(self, conf_name):
        try:
            Process.__init__(self)
            config = ConfigParser()
            config.read(conf_name)

            self.now_time = datetime.now()
            self.index_date = (self.now_time).strftime('%Y.%m.%d')
            self.set_logger(os.path.join(os.getcwd(), 'log')) #현재 위치에서 log
            self.branch_list = config.get("BRANCH", "BRANCH_LIST").split(',')

            ########################################################
            # Elastic 관련 정보
            ES_HOST = config.get("ELASTICSEARCH", "HOST")  # set es url or host
            ES_PORT = config.get("ELASTICSEARCH", "PORT")
            # ES_USER = config.get("ELASTICSEARCH", "USER")
            # ES_PW = config.get("ELASTICSEARCH", "PASSWD")
            #self.RAW_DATA_INDEX = config.get("ELASTICSEARCH", "INDEX_NAME") #기본 파싱 데이터 index
            self.train_index = config.get("ELASTICSEARCH", "TRAIN_INDEX")  #학습 데이터 index

            self.ES = Elasticsearch(
                [f"{ES_HOST}:{ES_PORT}"]
            )
            #실 서버용
            # self.ES = Elasticsearch(
            #     [f"{ES_HOST}:{ES_PORT}"],
            #     http_auth=(ES_USER, ES_PW),
            #     use_ssl=True,
            #     ca_certs=os.path.join(os.path.join(os.getcwd(), "elasticsearch-ca.pem"))
            # )
            #self.ES = Elasticsearch(hosts=ES_HOST, port=ES_PORT, http_auth=(ES_USER, ES_PW))
            ########################################################
            self.es_time_interval = int(config.get("OPTION", "TRAIN_TIME_INTERVAL"))
            self.model_scaler_path = config.get("PATH", "MODEL_SCALER_PATH")
            self.logger.info(f'Start_time: {self.now_time}')

        except Exception as ex:
            print(f'__init__ Error: {str(ex)}')

    def set_logger(self, log_path):
        if not os.path.exists(log_path):
            subprocess.getstatusoutput("mkdir -p " + log_path)

        self.logger = logging.getLogger("make_clf_model")

        self.logger.setLevel(logging.INFO)

        log_name = "make_clf_model.log"
        # formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(filename)s(%(lineno)d) %(message)s")
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] (%(lineno)d) %(message)s")
        file_handler = RotatingFileHandler(os.path.join(log_path, log_name), maxBytes=5 * 1024 * 1024, backupCount=10)
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def time_difference(self):
        try:
            now_time = datetime.now()
            self.logger.info(f"Run Time: {now_time - self.now_time} ")
        except Exception as ex:
            self.logger.critical(str(ex))

    def Request_Train_data(self, t1, t2):
        try:
            es_list = []
            # 실서버 연동 버전
            qry = '''
                        {
                          "query": {
                            "bool": {
                              "must": [
                                {
                                  "range":{
                                    "REG_DTTM": {
                                      "gte": "%s",
                                      "lt": "%s"
                                      }
                                  }
                                }
                              ]
                            }
                          }
                        }''' % (t1, t2)

            index_name = self.train_index + '*'
            es_result = self.ES.search(index=index_name, body=qry, size=30000)  # , scroll="1m")

            es_result_total = int(es_result['hits']['total']['value'])  # 검색결과 수
            if es_result_total == 0:
                self.logger.info(f"es_result ([UTC]{t1} ~ {t2}): No data")
                return []
            for obj in es_result['hits']['hits']:
                es_list.append(obj['_source'])
            # self.logger.info(f"es_result_{branch}({gte_str} ~ {lt_str}): ({len(es_list)}/{es_result_total})")
            self.logger.info(f"Train Data Request ([UTC]{t1} ~ {t2}): {len(es_list)}")
            return es_list  # [{dict}, {dict} ....]
        except Exception as ex:
            self.logger.critical("Request_Train_data Error")
            return []

    def Read_col_list(self):
        try:
            col_list_path = os.path.join(os.getcwd(), 'x_col_list')
            f = open(col_list_path, 'r')
            lines = f.readlines()
            new_line = [line.replace('\n','') for line in lines]
            f.close()
            return new_line
        except Exception as ex:
            self.logger.critical(f"Read_col_list Error: {ex}")

    def data_missing_value(self, es_list):
        try:
            train_df = pd.DataFrame(es_list)
            train_df = train_df.fillna(0)
            return train_df
        except Exception as ex:
            self.logger.critical(f"data_missing_value Error: {ex}")

    def make_oversampling(self, x, y, flag='ads', random_state=2022):
        try:
            if flag == 'ros':
                oversampling = RandomOverSampler(random_state=42)
            elif flag == 'smote':
                oversampling = SMOTE(random_state=42, k_neighbors=5)
            elif flag == 'bsmote':
                oversampling = BorderlineSMOTE(random_state=42, k_neighbors=5, m_neighbors=10)
            else: #ADASYN
                oversampling = ADASYN(random_state=42, n_neighbors=5)
            x_oversample, y_oversample = oversampling.fit_resample(x, y)

            return x_oversample, y_oversample
        except Exception as ex:
            self.logger.critical(f"make_oversampling Error: {ex}")

    def Make_Scaler(self, x_train, x_test, func='sds'):
        try:
            if func == 'mms':
                feature_scaler = MinMaxScaler()
                self.logger.info('MinMaxScaler')
            else:
                feature_scaler = StandardScaler()
                self.logger.info('StandardScaler')
            x_train = feature_scaler.fit_transform(x_train)
            x_test = feature_scaler.transform(x_test)

            #scaler 저장
            file_name = 'scaler.joblib'
            scaler_path = os.path.join(self.model_scaler_path, file_name)
            joblib.dump(feature_scaler, scaler_path)
            self.logger.info(f"Scaler Saved: {scaler_path}")

            return x_train, x_test
        except Exception as ex:
            self.logger.critical(f"Make_Scaler Error: {ex}")

    # train_test_split
    def Make_split_data(self, x, y, test_size=0.3, random_state=2021):
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
            self.logger.info(f'x_train.shape: {x_train.shape} | y_train.shape: {y_train.shape}')
            self.logger.info(f'x_test.shape: {x_test.shape} | y_test.shape: {y_test.shape}')
            return x_train, x_test, y_train, y_test
        except Exception as ex:
            self.logger.critical(f"Make_split_data Error: {ex}")

    def Make_Score_confusion_matrix(self, y_real, y_pred, model_NM='title'):
        try:
            file_path = os.path.join(self.model_scaler_path, model_NM + '.png')
            plt.figure(figsize=(10, 8))
            plt.title(model_NM)

            target_col_dict = {0.0: "Normal", 1.0: "Dongle_NW_Failure", 2.0: "POS_SW_Failure", 3.0: "Dongle_HW_Failure"}
            # target_col=["Normal", "Dongle_NW_Failure", "POS_SW_Failure", "Dongle_HW_Failure"]
            target_col = []
            for k, v in target_col_dict.items():
                if k in y_real:
                    target_col.append(v)

            cm = confusion_matrix(y_real, y_pred)
            sns.set(rc={'figure.figsize': (6, 6)})
            sns.heatmap(pd.DataFrame(cm, index=target_col, columns=target_col)[::-1], annot=True, fmt='d', annot_kws={"size": 15})
            plt.ylabel('Actual')
            plt.xlabel('Predict')
            plt.savefig(file_path)
            #plt.show()
            return file_path

        except Exception as ex:
            self.logger.critical(f"Make_Score_confusion_matrix Error: {ex}")

    def put_model(self, model_name, model_type, host, score1, score2, commnet, img_name, scaler_name):
        try:
            model_path = os.path.join(self.model_scaler_path, model_name+'.joblib')
            img_path = os.path.join(self.model_scaler_path, model_name+'.png')
            scaler_path = os.path.join(self.model_scaler_path, 'scaler.joblib')

            url = f"http://10.56.38.175/models/{model_name}"
            #model(model_000.joblib), result(result_000.png), scaler(scaler.joblib)
            files = [('model_'+model_name, open(model_path, "rb")), ('result_'+ img_name, open(img_path, "rb")),(scaler_name, open(scaler_path, "rb"))]
            data = {"model_name": model_name, "model_type":model_type, "host":host, "score1": score1, "score2": score2, "commnet": commnet}
            self.logger.info(f"files: {files}")
            self.logger.info(f"data: {data}")
            res = requests.put(url, data=data, files=files)
            self.logger.info(f"put_model result: {res.status_code}")
        except Exception as ex:
            self.logger.critical(f"put_model Error: {ex}")

    def main(self):
        #elastic에서 학습 데이터 쿼리: Request_Train_data()
        t1 = self.now_time - timedelta(days=self.es_time_interval)
        t2 = self.now_time
        es_list = self.Request_Train_data(t1.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:23], t2.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:23])

        #학습데이터 상태 검사 및 결측치 조정
        train_df = self.data_missing_value(es_list)

        #컬럼 리스트 불러오기
        x_col_list = self.Read_col_list()

        #오버샘플링 시행
        # 숫자 데이터 float 형태 변환
        x = train_df[x_col_list[1:]].values.astype(float)  # label필드(failure_category) 제외한 전부
        y = train_df['failure_category'].values.astype(float)  # label필드(failure_category)만
        y[-1] = 1.0
        y[-2] = 2.0
        y[-3] = 3.0
        X_oversample, y_oversample = self.make_oversampling(x, y, 'ros', 1988)

        # train_test_split
        x_train, x_test, y_train, y_test = self.Make_split_data(X_oversample, y_oversample, test_size=0.3, random_state=99)

        #Scaler 시행 / 저장
        x_train, x_test = self.Make_Scaler(x_train, x_test, func='mms')

        #모델 선언
        model_dicts = {
            'KNeighborsClassifier': set_clf_models.Set_KNeighborsClassifier(),
            'RandomForestClassifier': set_clf_models.Set_RandomForestClassifier(),
            'GaussianNB': set_clf_models.Set_GaussianNB(),
            'AdaBoostClassifier': set_clf_models.Set_AdaBoostClassifier(),
            'DecisionTreeClassifier': set_clf_models.Set_DecisionTreeClassifier(),
            'GradientBoostingClassifier': set_clf_models.Set_GradientBoostingClassifier(),
            'BaggingClassifier': set_clf_models.Set_BaggingClassifier(),
            'SVC': set_clf_models.Set_SVC()
        }
        #'LGBMClassifier': set_clf_models.Set_LGBMClassifier(),
        #'XGBClassifier_sk': Set_XGBClassifier_sk()
        model_dicts['voting'] = set_clf_models.Set_voting(model_dicts)  # voting 모델 추가 선언

        #모델 fit
        fit_models = {k: v.fit(x_train, y_train) for k, v in model_dicts.items()}
        self.logger.info('fit_done')

        #모델 예측
        pred_models = {k: v.predict(x_test) for k, v in fit_models.items()}
        self.logger.info('predict_done')

        #모델 저장
        for model_name, fit_model in fit_models.items():
            file_name = model_name +'.joblib'
            model_path = os.path.join(self.model_scaler_path, file_name)
            joblib.dump(fit_model, model_path)
            self.logger.info(f"{model_name} model saved ")

        # 모델평가(간단)
        model_f1 = {}
        model_acc = {}
        model_cm = {}
        for model_NM, model_pred in pred_models.items():
            # print(f"{model_NM} : {(round(accuracy_score(y_test, model_pred), 4))} | {(round(f1_score(y_test, model_pred, average='weighted'), 4))}" )
            model_acc[model_NM] = accuracy_score(y_test, model_pred)  # 모델별 정확도
            model_f1[model_NM] = f1_score(y_test, model_pred, average='weighted')  # 모델별 f1-score

            # confusion matrix
            model_cm[model_NM] = self.Make_Score_confusion_matrix(y_test, model_pred, model_NM)

            self.logger.info(f"{model_NM}:  acc({model_acc[model_NM]}) / f1({model_f1[model_NM]})")

            #self.put_model(model_name, model_type, host, score1, score2, commnet, img_name, scaler_name)
            self.put_model(model_NM, 'clf', 'none', model_acc[model_NM], model_f1[model_NM], 'test', model_NM, 'scaler')



if __name__ == '__main__':
    make_clf_model = MAKE_CLF_MODELS(os.path.join(os.getcwd(), 'config.ini'))
    make_clf_model.main()
    #make_clf_model.Read_col_list()
    make_clf_model.time_difference() #실행시간