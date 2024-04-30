import pandas as pd
from datasets import Dataset

def preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna('')

    # Transform into the desired dictionary format
    data_dict = {
        "prompt": df['prompt'].tolist(),
        "chosen": df['positive'].tolist(),
        "rejected": df['negative'].tolist()
    }

    return data_dict

def load_train_dataset(file_path):
    train_data_dict = preprocess(file_path)
    df = pd.DataFrame(train_data_dict)
    train_dataset = Dataset.from_pandas(df)
    return train_dataset
