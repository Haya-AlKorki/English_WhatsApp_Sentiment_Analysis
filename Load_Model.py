from sklearn.ensemble import GradientBoostingClassifier
import xgboost
import pickle

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

xgb = pickle.load(open('xgb.pkl', 'rb'))
# tokenizer_path = r'C:\Users\Haya\Desktop\Python\Grad_Project\tokenizer_distilbert\tokenizer_distil'
# model_path =r'C:\Users\Haya\Desktop\Python\Grad_Project\distilbert_model\distilbert_model'
#
#
# config = DistilBertConfig.from_json_file(r"C:\Users\Haya\Desktop\Python\Grad_Project\tokenizer_distilbert\tokenizer_distil\tokenizer_config.json")
# config2 = DistilBertConfig.from_json_file(r"C:\Users\Haya\Desktop\Python\Grad_Project\distilbert_model\distilbert_model\config.json")
#
# tokenizer_distil = DistilBertTokenizerFast(
#     vocab_file=r"C:\Users\Haya\Desktop\Python\Grad_Project\tokenizer_distilbert\tokenizer_distil\vocab.txt",
#     tokenizer_file=r"C:\Users\Haya\Desktop\Python\Grad_Project\tokenizer_distilbert\tokenizer_distil\tokenizer.json",
#     special_tokens_map_file=r"C:\Users\Haya\Desktop\Python\Grad_Project\tokenizer_distilbert\tokenizer_distil\special_tokens_map.json",
#     config=config,
#     local_files_only=True,
# )
#
# distilbert = TFDistilBertForSequenceClassification.from_pretrained(
#     model_path,
#     config=config2,
#     local_files_only=True
# )
