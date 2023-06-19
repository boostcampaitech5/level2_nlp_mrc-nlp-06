from train_dpr import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, default="./data/train_dataset", help=""
    )
    
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        default ="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, default="./data", help="")
    
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, default = "wikipedia_documents.json", help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, default = False, help="")

    args = parser.parse_args()
    
    
    model_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01
    )
    
    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    train_dataset = org_dataset["train"].flatten_indices()
    valid_dataset = org_dataset["validation"].flatten_indices()

    if model_args.device.type!='cuda':
        print('gpu 사용 불가')
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 학습된 encoder 가져오기
    q_encoder = BertEncoder.from_pretrained(args.model_name_or_path).to(device)
    p_encoder = BertEncoder.from_pretrained(args.model_name_or_path).to(device)
    p_encoder.load_state_dict(torch.load('./src/retriever/passage_encoder/model.pt'))
    q_encoder.load_state_dict(torch.load('./src/retriever/query_encoder/model.pt'))
    
    
    retriever = DenseRetrieval(model_name = 'klue/bert-base',
                           dataset = org_dataset,
                           q_encoder = q_encoder,
                           p_encoder = p_encoder,
                           args = model_args)
    
    df = pd.DataFrame()
    df = retriever.retrieve(valid_dataset,100, 'dense_embedding')
    
    df.to_csv('./src/retriever/retrieval_result/dense.csv',index = False)