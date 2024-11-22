import numpy as np
from numpy import ndarray

from typing import List

import torch
from transformers import AutoTokenizer,AutoModel
from sympy.integrals.meijerint_doc import category
from torch.utils.tensorboard.summary import image

from model.compatibility.src.models.model_args import Args
from model.compatibility.src.datasets.processor import FashionInputProcessor,FashionImageProcessor
from model.compatibility.src.models.embedder import  OutfitTransformerEmbeddingModel
from model.compatibility.src.models.recommender import RecommendationModel

#model 정보 가지고 오기
def load_model(args):
    image_processor = FashionImageProcessor()
    text_tokenizer = AutoTokenizer.from_pretrained(args.huggingface)

    input_processor = FashionInputProcessor(
        categories=args.categories,
        use_image=args.use_image,
        image_processor=image_processor,
        use_text=args.use_text,
        text_tokenizer=text_tokenizer,
        text_max_length=args.text_max_length,
        text_padding='max_length',
        text_truncation=True,
        outfit_max_length=args.outfit_max_length
    )

    embedding_model = OutfitTransformerEmbeddingModel(
        input_processor=input_processor,
        hidden=args.hidden,
        huggingface=args.huggingface,
        normalize=args.normalize
    )

    recommendation_model = RecommendationModel(
        embedding_model=embedding_model,
        ffn_hidden=args.ffn_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    if args.load_model:
        checkpoint = torch.load(args.model_path, map_location='cuda')
        state_dict = checkpoint['state_dict']
        recommendation_model.load_state_dict(state_dict)
        print(f'[COMPLETE] Load from {args.model_path}')

    return recommendation_model, input_processor

def make_embedding(image: np.ndarray, description: str, category : str, model, input_processor) -> dict:
    # args = Args()
    # args.model_path = './model/compatibility/src/checkpoints/cp_auc0.91.pth'
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # model, input_processor = load_model(args)
    # model.to(device)
    #
    # model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            inputs = input_processor(category, image, texts=description)
            inputs = { key: torch.unsqueeze(value,0).to(device) for key, value in inputs.items()}
            input_embeddings = model.batch_encode(inputs)

    return input_embeddings


def make_score(embeddings: List[dict],model, input_processor) -> int:
    # args = Args()
    # args.model_path = './model/compatibility/src/checkpoints/cp_auc0.91.pth'
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # model, input_processor = load_model(args)
    # model.to(device)
    #
    # model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            mask = []
            embed = []
            for embedding in embeddings:
                mask.append(embedding['mask'])
                embed.append(embedding['embeds'])

            input_embeddings = {'mask' : torch.cat(mask, dim=1), 'embeds' : torch.cat(embed, dim=1)}

            probs = model.get_score(input_embeddings)

        score = probs.flatten().detach().cpu().tolist()[0]
        print(f"score : {int(score*100)}")
    return int(score*100)

