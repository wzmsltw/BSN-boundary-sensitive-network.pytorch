import sys
sys.dont_write_bytecode = True
import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import opts
from dataset import VideoDataSet,ProposalDataSet
from models import TEM,PEM
from loss_function import TEM_loss_function,PEM_loss_function
import pandas as pd
from pgm import PGM_proposal_generation,PGM_feature_generation
from post_processing import BSN_post_processing
from eval import evaluation_proposal

def train_TEM(data_loader,model,optimizer,epoch,writer,opt):
    model.train()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_cost = 0
    for n_iter,(input_data,label_action,label_start,label_end) in enumerate(data_loader):
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action,label_start,label_end,TEM_output,opt)
        cost = loss["cost"] 
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        epoch_action_loss += loss["loss_action"].cpu().detach().numpy()
        epoch_start_loss += loss["loss_start"].cpu().detach().numpy()
        epoch_end_loss += loss["loss_end"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()
    
    writer.add_scalars('data/action', {'train': epoch_action_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/start', {'train': epoch_start_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/end', {'train': epoch_end_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/cost', {'train': epoch_cost/(n_iter+1)}, epoch)

    print "TEM training loss(epoch %d): action - %.03f, start - %.03f, end - %.03f" %(epoch,epoch_action_loss/(n_iter+1),
                                                                                        epoch_start_loss/(n_iter+1),
                                                                                        epoch_end_loss/(n_iter+1))

def test_TEM(data_loader,model,epoch,writer,opt):
    model.eval()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_cost = 0
    for n_iter,(input_data,label_action,label_start,label_end) in enumerate(data_loader):
        
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action,label_start,label_end,TEM_output,opt)
        epoch_action_loss += loss["loss_action"].cpu().detach().numpy()
        epoch_start_loss += loss["loss_start"].cpu().detach().numpy()
        epoch_end_loss += loss["loss_end"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()
    
    writer.add_scalars('data/action', {'test': epoch_action_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/start', {'test': epoch_start_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/end', {'test': epoch_end_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/cost', {'test': epoch_cost/(n_iter+1)}, epoch)
    
    print "TEM testing  loss(epoch %d): action - %.03f, start - %.03f, end - %.03f" %(epoch,epoch_action_loss/(n_iter+1),
                                                                                        epoch_start_loss/(n_iter+1),
                                                                                        epoch_end_loss/(n_iter+1))
    state = {'epoch': epoch + 1,
                'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"]+"/tem_checkpoint.pth.tar" )
    if epoch_cost< model.module.tem_best_loss:
        model.module.tem_best_loss = np.mean(epoch_cost)
        torch.save(state, opt["checkpoint_path"]+"/tem_best.pth.tar" )

def train_PEM(data_loader,model,optimizer,epoch,writer,opt):
    model.train()
    epoch_iou_loss = 0
    
    for n_iter,(input_data,label_iou) in enumerate(data_loader):
        PEM_output = model(input_data)
        iou_loss = PEM_loss_function(PEM_output,label_iou,model,opt)
        optimizer.zero_grad()
        iou_loss.backward()
        optimizer.step()
        epoch_iou_loss += iou_loss.cpu().detach().numpy()

    writer.add_scalars('data/iou_loss', {'train': epoch_iou_loss/(n_iter+1)}, epoch)
    
    print "PEM training loss(epoch %d): iou - %.04f" %(epoch,epoch_iou_loss/(n_iter+1))

def test_PEM(data_loader,model,epoch,writer,opt):
    model.eval()
    epoch_iou_loss = 0
    
    for n_iter,(input_data,label_iou) in enumerate(data_loader):
        PEM_output = model(input_data)
        iou_loss = PEM_loss_function(PEM_output,label_iou,model,opt)
        epoch_iou_loss += iou_loss.cpu().detach().numpy()

    writer.add_scalars('data/iou_loss', {'validation': epoch_iou_loss/(n_iter+1)}, epoch)
    
    print "PEM testing  loss(epoch %d): iou - %.04f" %(epoch,epoch_iou_loss/(n_iter+1))
    
    state = {'epoch': epoch + 1,
                'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"]+"/pem_checkpoint.pth.tar" )
    if epoch_iou_loss<model.module.pem_best_loss :
        model.module.pem_best_loss = np.mean(epoch_iou_loss)
        torch.save(state, opt["checkpoint_path"]+"/pem_best.pth.tar" )


def BSN_Train_TEM(opt):
    writer = SummaryWriter()
    model = TEM(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=opt["tem_training_lr"],weight_decay = opt["tem_weight_decay"])
    
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="train"),
                                                batch_size=model.module.batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True,drop_last=True)            
    
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="validation"),
                                                batch_size=model.module.batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True,drop_last=True)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["tem_step_size"], gamma = opt["tem_step_gamma"])
        
    for epoch in range(opt["tem_epoch"]):
        scheduler.step()
        train_TEM(train_loader,model,optimizer,epoch,writer,opt)
        test_TEM(test_loader,model,epoch,writer,opt)
    writer.close()
    


def BSN_Train_PEM(opt):
    writer = SummaryWriter()
    model = PEM(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=opt["pem_training_lr"],weight_decay = opt["pem_weight_decay"])
    
    def collate_fn(batch):
        batch_data = torch.cat([x[0] for x in batch])
        batch_iou = torch.cat([x[1] for x in batch])
        return batch_data,batch_iou
    
    train_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset="train"),
                                                batch_size=model.module.batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True,drop_last=True,collate_fn=collate_fn)            
    
    test_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset="validation"),
                                                batch_size=model.module.batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True,drop_last=True,collate_fn=collate_fn)
        
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["pem_step_size"], gamma = opt["pem_step_gamma"])
        
    for epoch in range(opt["pem_epoch"]):
        scheduler.step()
        train_PEM(train_loader,model,optimizer,epoch,writer,opt)
        test_PEM(test_loader,model,epoch,writer,opt)
        
    writer.close()


def BSN_inference_TEM(opt):
    model = TEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"]+"/tem_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()
    
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="full"),
                                                batch_size=model.module.batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True,drop_last=False)
    
    columns=["action","start","end","xmin","xmax"]
    for index_list,input_data,anchor_xmin,anchor_xmax in test_loader:
        
        TEM_output = model(input_data).detach().cpu().numpy()
        batch_action = TEM_output[:,0,:]
        batch_start = TEM_output[:,1,:]
        batch_end = TEM_output[:,2,:]
        
        index_list = index_list.numpy()
        anchor_xmin = np.array([x.numpy()[0] for x in anchor_xmin])
        anchor_xmax = np.array([x.numpy()[0] for x in anchor_xmax])
        
        for batch_idx,full_idx in enumerate(index_list):            
            video = test_loader.dataset.video_list[full_idx]
            video_action = batch_action[batch_idx]
            video_start = batch_start[batch_idx]
            video_end = batch_end[batch_idx]    
            video_result = np.stack((video_action,video_start,video_end,anchor_xmin,anchor_xmax),axis=1)
            video_df = pd.DataFrame(video_result,columns=columns)  
            video_df.to_csv("./output/TEM_results/"+video+".csv",index=False)
            
            
def BSN_inference_PEM(opt):
    model = PEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"]+"/pem_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()
    
    test_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset=opt["pem_inference_subset"]),
                                                batch_size=model.module.batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True,drop_last=False)
    
    for idx,(video_feature,video_xmin,video_xmax,video_xmin_score,video_xmax_score) in enumerate(test_loader):
        video_name = test_loader.dataset.video_list[idx]
        video_conf = model(video_feature).view(-1).detach().cpu().numpy()
        video_xmin = video_xmin.view(-1).cpu().numpy()
        video_xmax = video_xmax.view(-1).cpu().numpy()
        video_xmin_score = video_xmin_score.view(-1).cpu().numpy()
        video_xmax_score = video_xmax_score.view(-1).cpu().numpy()
        
        df=pd.DataFrame()
        df["xmin"]=video_xmin
        df["xmax"]=video_xmax
        df["xmin_score"]=video_xmin_score
        df["xmax_score"]=video_xmax_score
        df["iou_score"]=video_conf
        
        df.to_csv("./output/PEM_results/"+video_name+".csv",index=False)


def main(opt):
    if opt["module"] == "TEM":
        if opt["mode"] == "train":
            print "TEM training start"  
            BSN_Train_TEM(opt)
            print "TEM training finished"  
        elif opt["mode"] == "inference":
            print "TEM inference start"  
            if not os.path.exists("output/TEM_results"):
                os.makedirs("output/TEM_results") 
            BSN_inference_TEM(opt)
            print "TEM inference finished"
        else:
            print "Wrong mode. TEM has two modes: train and inference"
          
    elif opt["module"] == "PGM":
        if not os.path.exists("output/PGM_proposals"):
            os.makedirs("output/PGM_proposals") 
        print "PGM: start generate proposals"
        PGM_proposal_generation(opt)
        print "PGM: finish generate proposals"
        
        if not os.path.exists("output/PGM_feature"):
            os.makedirs("output/PGM_feature") 
        print "PGM: start generate BSP feature"
        PGM_feature_generation(opt)
        print "PGM: finish generate BSP feature"
    
    elif opt["module"] == "PEM":
        if opt["mode"] == "train":
            print "PEM training start"  
            BSN_Train_PEM(opt)
            print "PEM training finished"  
        elif opt["mode"] == "inference":
            if not os.path.exists("output/PEM_results"):
                os.makedirs("output/PEM_results") 
            print "PEM inference start"  
            BSN_inference_PEM(opt)
            print "PEM inference finished"
        else:
            print "Wrong mode. PEM has two modes: train and inference"
    
    elif opt["module"] == "Post_processing":
        print "Post processing start"
        BSN_post_processing(opt)
        print "Post processing finished"
        
    elif opt["module"] == "Evaluation":
        evaluation_proposal(opt)
    print ""
        
if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"]) 
    opt_file=open(opt["checkpoint_path"]+"/opts.json","w")
    json.dump(opt,opt_file)
    opt_file.close()
    main(opt)
