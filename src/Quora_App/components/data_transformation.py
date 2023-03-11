from Quora_App.logger import logging
from Quora_App.exception import ApplicationException
from Quora_App.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from Quora_App.entity import DataTransformationArtifact, DataIngestionArtifact
from Quora_App.config import ConfigurationManager
from Quora_App.constants import *
from Quora_App.utils import *
from tqdm import tqdm
from datetime import datetime
import os, sys
import re
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import distance
import fuzzywuzzy as fuzz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self):
        logging.info(f"\n{'*'*5} Feature Engineering Started {'*'*5}\n")
        nltk.download('stopwords')
        self.SAFE_DIV = 0.0001 # to get result in 4 decimal points
        self.STOP_WORDS = stopwords.words('english')

    def fit(self,X,y=None):
        return self

    def transform(self, X,y=None):
        try:
            print(f"{'*'*5}Transformation Started{'*'*5}")
            
            logging.info(f"-----> Data Cleaning in process{'.'*10}")
            X = self.data_cleaning(X)
            
            # preprocessing each question
            logging.info(f"-----> Questions Text Cleaning Started{'.'*10}")
            X['question1'] = X["question1"].fillna("").apply(self.preprocess)
            X['question2'] = X["question2"].fillna("").apply(self.preprocess)
            
            df = self.common_features(self.extract_features(X))
            df = df.drop(columns=["qid1","qid2"])
            
            logging.info(f"-----> Finalizing DataFrame{'.'*10}")
            return df
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def data_cleaning(self,data):
        try:
            data.fillna('', inplace=True)
            indexes=[]
            pattern = '''[!"#%&'(\*)\+,\./:;<=>\?@\[\]\^_`\{\|\}~]'''
            for row in data.iterrows():
                r_id = row[1].id
                q1 = row[1].question1.lower()
                q2 = row[1].question2.lower()
                
                if len(q1) < 10:
                    indexes.append(r_id)
                q1 = re.sub(pattern, '',q1)
                
                if q1 in ['deleted','removed']:
                    indexes.append(r_id)
                    
                if len(q2) < 10:
                    indexes.append(r_id)
                q2 = re.sub(pattern, '',q2)
                
                if q2 in ['deleted','removed']:
                    indexes.append(r_id)
                    
            data.drop(index=indexes, inplace=True)
            return data
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def normalized_word_Common(self,row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
        return 1.0 * len(w1 & w2)


    def normalized_word_Total(self, row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
        return 1.0*(len(w1) + len(w2))


    def normalized_word_share(self, row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

    def common_features(self, data:pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info(f"-----> Finding Common Features{'.'*10}")
            data['q1len'] = data['question1'].str.len()
            data['q2len'] = data['question2'].str.len()
            data['q1_n_words'] = data['question1'].apply(lambda row: len(row.split(" ")))
            data['q2_n_words'] = data['question2'].apply(lambda row: len(row.split(" ")))
            data['word_Common'] = data.apply(self.normalized_word_Common, axis=1)
            data['word_Total'] = data.apply(self.normalized_word_Total, axis=1)
            data['word_share'] = data.apply(self.normalized_word_share, axis=1)
            return data
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def preprocess(self, x):
        try:
            x = str(x).lower()
            x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                                .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                                .replace("n't", " not").replace("what's", "what is").replace("it's", "it is").replace("didn't","did not")\
                                .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                                .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                                .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                                .replace("€", " euro ").replace("'ll", " will")
            x = re.sub(r"([0-9]+)000000", r"\1m", x)
            x = re.sub(r"([0-9]+)000", r"\1k", x)

            porter = PorterStemmer()
            pattern1 = re.compile('\W')

            if type(x) == type(''):
                x = re.sub(pattern1,' ',x)

            if type(x) == type(''):
                x = porter.stem(x)
                example1 = BeautifulSoup(x)
                x = example1.get_text()
                
            ## removing remaining punctuations
            pattern2 = '''[!"#%&'(\*)\+,\./:;<=>\?@\[\]\^_`\{\|\}~]'''
            x= re.sub(pattern2, '',x).strip()
            return x
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def get_token_features(self, q1, q2):
        try:
            token_features = [0.0]*10

            # Converting questions into tokens
            q1_tokens = q1.split()
            q2_tokens = q2.split()

            if len(q1_tokens)== 0 or len(q2_tokens)==0:
                return token_features

            # Getting non-stopwords in Questions
            q1_words = set([word for word in q1_tokens if word not in self.STOP_WORDS])
            q2_words = set([word for word in q2_tokens if word not in self.STOP_WORDS])

            # Getting stopwords in Questions
            q1_stops = set([word for word in q1_tokens if word in self.STOP_WORDS])
            q2_stops = set([word for word in q2_tokens if word in self.STOP_WORDS])

            # Getting the common non-stopwords from questions pair
            common_word_count = len(set(q1_words).intersection(set(q2_words)))

            # Getting the common stopwords from questions pair
            common_stop_count = len(set(q1_stops).intersection(set(q2_stops)))

            # Getting the common non-stopwords from questions pair
            common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

            token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + self.SAFE_DIV)
            token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + self.SAFE_DIV)
            token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
            token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
            token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)
            token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)

            # Last word of both questions is same or not
            token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

            # First word of both questions is same or not
            token_features[7] = int(q1_tokens[0] == q2_tokens[0])

            token_features[8] = abs(len(q1_tokens) - len(q2_tokens))

            # Average token length of both Questions
            token_features[9] = (len(q1_tokens) + len(q2_tokens))/2

            return token_features
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def get_longest_substr_ratio(self,a,b):
        strs = list(distance.lcsubstrings(a,b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b))+1)

    def extract_features(self,df):
        try:
            logging.info(f"-----> Finding Token Features{'.'*10}")

            # Merging features with dataset
            token_features = df.apply(lambda x: self.get_token_features(x['question1'], x['question2']), axis=1)

            df["cwc_min"]       = list(map(lambda x:x[0], token_features))
            df["cwc_max"]       = list(map(lambda x:x[1], token_features))
            df["csc_min"]       = list(map(lambda x:x[2], token_features))
            df["csc_max"]       = list(map(lambda x:x[3], token_features))
            df["ctc_min"]       = list(map(lambda x:x[4], token_features))
            df["ctc_max"]       = list(map(lambda x:x[5], token_features))
            df["last_word_eq"]  = list(map(lambda x:x[6], token_features))
            df["first_word_eq"] = list(map(lambda x:x[7], token_features))
            df["abs_len_diff"]  = list(map(lambda x:x[8], token_features))
            df["mean_len"]      = list(map(lambda x:x[9], token_features))

            ## Computing fuzzy features
            print(f"-----> Finding Fuzzy Features{'.'*10}")

            df['token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(x['question1'], x['question2']),axis=1)
            df['token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(x['question1'], x['question2']),axis=1)
            df['fuzz_ratio'] = df.apply(lambda x: fuzz.ratio(x['question1'], x['question2']),axis=1)
            df['q_ratio'] = df.apply(lambda x: fuzz.QRatio(x['question1'], x['question2']),axis=1)
            df['W_ratio'] = df.apply(lambda x: fuzz.WRatio(x['question1'], x['question2']),axis=1)
            df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(x['question1'], x['question2']),axis=1)
            df['longest_substr_ratio'] = df.apply(lambda x: self.get_longest_substr_ratio(x['question1'], x['question2']),axis=1)
            return df
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def tfidf_vectorizer(self,df:pd.DataFrame)->dict:
        logging.info(f"-----> Finding TF-IDF Vectors{'.'*10}")
        try:
            ## Merging texts of questions
            questions = list(df['question1'] + df['question2'])

            ## Finding tfidf of questions
            tfidf = TfidfVectorizer(lowercase=False)
            tfidf.fit_transform(questions)

            # dict key:word and value:tf-idf score
            word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
            return word2tfidf
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def vec_generator(self, df:pd.DataFrame, feature:str, word2tfidf:dict, nlp)->list:
        try:
            logging.info(f"-----> Generating sentence vectors for {feature} {'.'*10}")
            vecs = []
            for qu in tqdm(list(df[feature])):
                doc = nlp(qu)
                mean_vec = np.zeros([len(doc), len(doc[0].vector)]) 
                for word in doc:
                    vec = word.vector
                    try:
                        idf = word2tfidf[str(word)]
                    except:
                        idf = 0
                    mean_vec += vec*idf
                mean_vec = mean_vec.mean(axis=0)
                vecs.append(mean_vec)
            return list(vecs)
        except Exception as e:
            raise ApplicationException(e, sys) from e
    
    def similarity_distance_calculate(self,q1,q2):
        try:
            cosine_sim=[]
            euclidean_dist=[]
            for idx in tqdm(range(len(q1))):
                cosine_sim.append(cosine_similarity([q1[idx]],[q2[idx]]))
                euclidean_dist.append(euclidean_distances([q1[idx]],[q2[idx]]))

            dist_df = pd.DataFrame({"cosine_sim":np.array(cosine_sim).flatten(),
                                    "euclidean_dist":np.array(euclidean_dist).flatten()})
            return dist_df
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def __del__(self):
        logging.info(f"\n{'*'*5} Feature Engineering completed Successfully {'*'*5}")
        


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                        data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"\n{'>'*20} Data Transformation Started {'<'*20}\n")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.start = datetime.now()
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            logging.info("Reading data from root dir")
            raw_data_dir = self.data_ingestion_artifact.data_file_path
            data= pd.DataFrame()
            for file in os.listdir(raw_data_dir):
                temp_df= read_data(Path(raw_data_dir,file))
                data= pd.concat([data, temp_df], axis=0)
            logging.info("Data read successfully")

            logging.info("Feature Engineering object")
            feat_eng = FeatureEngineering()
            transformed_df = feat_eng.fit_transform(data)

            word2tfidf = feat_eng.tfidf_vectorizer(transformed_df)

            nlp = spacy.load("en_core_web_lg")

            transformed_df["q1_feats_m"]= feat_eng.vec_generator(transformed_df, "question1",word2tfidf, nlp)
            transformed_df["q2_feats_m"]= feat_eng.vec_generator(transformed_df, "question2",word2tfidf,nlp)

            logging.info(f"-----> Processing Vectors generated{'.'*10}")
            q1 = pd.DataFrame(transformed_df['q1_feats_m'].values.tolist(), index = transformed_df.index)
            q2 = pd.DataFrame(transformed_df['q2_feats_m'].values.tolist(), index = transformed_df.index)

            dist_df=feat_eng.similarity_distance_calculate(transformed_df["q1_feats_m"].values.tolist(),
                                           transformed_df["q2_feats_m"].values.tolist())
            
            logging.info("Preparing Final Transformed Dataframe......")
            transformed_df.drop(['question1','question2','q1_feats_m','q2_feats_m'],axis=1, inplace=True)
            q1['id']=transformed_df['id']
            q2['id']=transformed_df['id']
            dist_df['id']=transformed_df['id']
            transformed_df = transformed_df.merge(dist_df, on='id',how='left')
            df2  = q1.merge(q2, on='id',how='left')
            final_df  = transformed_df.merge(df2, on='id',how='left')
            final_df.dropna(axis=0, inplace=True)

            logging.info("Splitting Data into train and test")
            y_true= final_df['is_duplicate']
            # X= final_df.drop(['id','is_duplicate'], axis=1)
            train, test= train_test_split(final_df, stratify=y_true, test_size=0.20, random_state=30)

            logging.info("Scaling feature 'euclidean dist'.....")
            sc = StandardScaler()
            train['euclidean_dist'] = sc.fit_transform(train[['euclidean_dist']])
            test['euclidean_dist'] = sc.transform(test[['euclidean_dist']])

            logging.info("Saving train data")
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            create_directories([transformed_train_dir])
            transformed_train_file_path= Path(transformed_train_dir, "transformed_train.parquet")
            save_data(filepath=transformed_train_file_path, df=train, format='parquet')

            logging.info("Saving test data")
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            create_directories([transformed_test_dir])
            transformed_test_file_path= Path(transformed_test_dir, "transformed_test.parquet")
            save_data(filepath=transformed_test_file_path, df=test, format='parquet')

            logging.info("Saving feature engineering object")
            feat_eng_obj_file_path= self.data_transformation_config.feature_eng_object_file_path
            create_directories([os.path.dirname(feat_eng_obj_file_path)])
            save_bin(data=feat_eng, path=feat_eng_obj_file_path)

            logging.info("Saving word2tfidf object")
            word2tfidf_object_file_path= self.data_transformation_config.word2tfidf_object_file_path
            save_bin(data=word2tfidf, path=word2tfidf_object_file_path)

            logging.info("Saving scaler object")
            preprocessed_obj_file_path= self.data_transformation_config.preprocessed_object_file_path
            save_bin(data=sc, path=preprocessed_obj_file_path)

            data_transformation_artifacts= DataTransformationArtifact(is_transformed=True,
                                                        transformed_train_file_path=transformed_train_file_path,
                                                        transformed_test_file_path=transformed_test_file_path,
                                                        feat_eng_obj_file_path=feat_eng_obj_file_path,
                                                        preprocessed_obj_file_path=preprocessed_obj_file_path,
                                                        word2tfidf_object_file_path=word2tfidf_object_file_path)
            return data_transformation_artifacts

        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def __init__(self):
        logging.info(f"\n{'>'*20} Data Transformation Completed {'<'*20}\n")
        logging.info(f"{'>'*5} Time taken for Execution : {datetime.now()-self.start} {'<'*5}\n")

    
    
