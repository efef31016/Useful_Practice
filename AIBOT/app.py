from flask import Flask, render_template, request, jsonify
from ChatBotModel import BotModel
import torch
from torch import nn

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    
    user_message = request.json['message']
    
    # NLP 模型
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    # Configure models
    attn_model = 'dot'
    hidden_size = 50
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    # load
    loadFilename = "2000_checkpoint.tar"
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc_dict = checkpoint['voc_dict']
    # model
    embedding = nn.Embedding(len(voc_dict), hidden_size)
    embedding.load_state_dict(embedding_sd)
    encoder = BotModel.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = BotModel.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, len(voc_dict), decoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    response =  BotModel.AiBot(user_message, encoder, decoder, voc_dict)
    return jsonify({'message': response})

