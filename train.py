from tensorflow.keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler,TensorBoard
from tensorflow.keras.models import load_model
import random
import numpy as np
from scipy import misc
import gc
from tensorflow.keras.optimizers import Adam
from imageio import imread
from datetime import datetime
import os
import json
import models
from utils import DataLoader, LrPolicy
from config import Config
import argparse
import glob

def get_parser():
    
    parser = argparse.ArgumentParser('train')
    parser.add_argument('--configPath', '-c', required=True)
    parser.add_argument('--num_train', type=int, help='num_train_dataset')
    return parser

def train(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    conf=Config()
    conf.load(args.configPath)
    time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    trainString="%s_%s_%s_%s" % (conf.model,conf.optimizer,str(conf.lr),time)
    
    logPath = f"{conf.logPath}_numTrain_{args.num_train}"
    
    os.makedirs(logPath+"/"+trainString)
    conf.save(logPath+"/"+trainString+'/config.json')
    print('Compiling model...')

    model_checkpoint = ModelCheckpoint(logPath+"/"+trainString+'/Checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=False, save_weights_only=True)
    change_lr = LearningRateScheduler(LrPolicy(conf.lr).stepDecay)
    tbCallBack=TensorBoard(log_dir=logPath+"/"+trainString+'/logs', histogram_freq=0,  write_graph=True, write_images=True)
    model=models.modelCreator(conf.model,conf.inputShape,conf.classes,conf.pretrainedModel)
    model.compile(optimizer = conf.optimizer, loss = conf.loss)
    # data = [conf.trainDataPath+"/"+f for f in os.listdir(conf.trainDataPath) if '.png' in f]
    data = glob.glob(os.path.join(conf.trainDataPath, "*.png"))
    random.seed(1)
    random.shuffle(data)
    # thr=int(len(data)*conf.validationSplit)
    thr = int(args.num_train)

    trainData=data[thr:]
    valData=data[:thr]
    trainDataLoader=DataLoader(conf.batchSize,conf.inputShape,trainData,conf.guaMaxValue)
    validationDataLoader=DataLoader(conf.batchSize,conf.inputShape,valData,conf.guaMaxValue)
    print('Fitting model...')
    model.fit_generator(generator=trainDataLoader.generator(),
                    validation_data=validationDataLoader.generator(),
                    steps_per_epoch=len(trainData)//conf.batchSize,
                    validation_steps=len(valData)//conf.batchSize,
                    epochs=conf.epoches,
                    verbose=1,
                    initial_epoch=0,
                    callbacks = [model_checkpoint, change_lr, tbCallBack]
                    )

if __name__ == "__main__":
   train()
