# -*- coding: utf-8 -*-
import json
import numpy
import pandas
import torch.multiprocessing as mp
from scipy.interpolate import interp1d

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data
    
def iou_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = numpy.maximum(anchors_min, box_min)
    int_xmax = numpy.minimum(anchors_max, box_max)
    inter_len = numpy.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len +box_max-box_min
    jaccard = numpy.divide(inter_len, union_len)
    return jaccard

def ioa_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = numpy.maximum(anchors_min, box_min)
    int_xmax = numpy.minimum(anchors_max, box_max)
    inter_len = numpy.maximum(int_xmax - int_xmin, 0.)
    scores = numpy.divide(inter_len, len_anchors)
    return scores

def generateProposals(opt,video_list,video_dict):
    tscale = opt["temporal_scale"]    
    tgap = 1./tscale
    peak_thres= opt["pgm_threshold"]

    for video_name in video_list:
        tdf=pandas.read_csv("./output/TEM_results/"+video_name+".csv")
        start_scores=tdf.start.values[:]
        end_scores=tdf.end.values[:]
        
        max_start = max(start_scores)
        max_end = max(end_scores)
        
        start_bins=numpy.zeros(len(start_scores))
        start_bins[[0,-1]]=1
        for idx in range(1,tscale-1):
            if start_scores[idx]>start_scores[idx+1] and start_scores[idx]>start_scores[idx-1]:
                start_bins[idx]=1
            elif start_scores[idx]>(peak_thres*max_start):
                start_bins[idx]=1
                    
        end_bins=numpy.zeros(len(end_scores))
        end_bins[[0,-1]]=1
        for idx in range(1,tscale-1):
            if end_scores[idx]>end_scores[idx+1] and end_scores[idx]>end_scores[idx-1]:
                end_bins[idx]=1
            elif end_scores[idx]>(peak_thres*max_end):
                end_bins[idx]=1
        
        xmin_list=[]
        xmin_score_list=[]
        xmax_list=[]
        xmax_score_list=[]
        for j in range(tscale):
            if start_bins[j]==1:
                xmin_list.append(tgap/2+tgap*j)
                xmin_score_list.append(start_scores[j])
            if end_bins[j]==1:
                xmax_list.append(tgap/2+tgap*j)
                xmax_score_list.append(end_scores[j])
                
        new_props=[]
        for ii in range(len(xmax_list)):
            tmp_xmax=xmax_list[ii]
            tmp_xmax_score=xmax_score_list[ii]
            
            for ij in range(len(xmin_list)):
                tmp_xmin=xmin_list[ij]
                tmp_xmin_score=xmin_score_list[ij]
                if tmp_xmin>=tmp_xmax:
                    break
                new_props.append([tmp_xmin,tmp_xmax,tmp_xmin_score,tmp_xmax_score])
        new_props=numpy.stack(new_props)
        
        col_name=["xmin","xmax","xmin_score","xmax_score"]
        new_df=pandas.DataFrame(new_props,columns=col_name)  
        new_df["score"]=new_df.xmin_score*new_df.xmax_score
        
        new_df=new_df.sort_values(by="score",ascending=False)
        
        video_info=video_dict[video_name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        corrected_second=float(feature_frame)/video_frame*video_second
        
        try:
            gt_xmins=[]
            gt_xmaxs=[]
            for idx in range(len(video_info["annotations"])):
                gt_xmins.append(video_info["annotations"][idx]["segment"][0]/corrected_second)
                gt_xmaxs.append(video_info["annotations"][idx]["segment"][1]/corrected_second)
            new_iou_list=[]
            for j in range(len(new_df)):
                tmp_new_iou=max(iou_with_anchors(new_df.xmin.values[j],new_df.xmax.values[j],gt_xmins,gt_xmaxs))
                new_iou_list.append(tmp_new_iou)
                
            new_ioa_list=[]
            for j in range(len(new_df)):
                tmp_new_ioa=max(ioa_with_anchors(new_df.xmin.values[j],new_df.xmax.values[j],gt_xmins,gt_xmaxs))
                new_ioa_list.append(tmp_new_ioa)
            new_df["match_iou"]=new_iou_list
            new_df["match_ioa"]=new_ioa_list
        except:
            pass
        new_df.to_csv("./output/PGM_proposals/"+video_name+".csv",index=False)


def getDatasetDict(opt):
    df=pandas.read_csv(opt["video_info"])
    json_data= load_json(opt["video_anno"])
    database=json_data
    video_dict = {}
    for i in range(len(df)):
        video_name=df.video.values[i]
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_new_info['annotations']=video_info['annotations']
        video_new_info['subset'] = df.subset.values[i]
        video_dict[video_name]=video_new_info
    return video_dict

def generateFeature(opt,video_list,video_dict):

    num_sample_start=opt["num_sample_start"]
    num_sample_end=opt["num_sample_end"]
    num_sample_action=opt["num_sample_action"]
    num_sample_interpld = opt["num_sample_interpld"]

    for video_name in video_list:
        adf=pandas.read_csv("./output/TEM_results/"+video_name+".csv")
        score_action=adf.action.values[:]
        seg_xmins = adf.xmin.values[:]
        seg_xmaxs = adf.xmax.values[:]
        video_scale = len(adf)
        video_gap = seg_xmaxs[0] - seg_xmins[0]
        video_extend = video_scale / 4 + 10
        pdf=pandas.read_csv("./output/PGM_proposals/"+video_name+".csv")
        video_subset = video_dict[video_name]['subset']
        if video_subset == "training":
            pdf=pdf[:opt["pem_top_K"]]
        else:
            pdf=pdf[:opt["pem_top_K_inference"]]
        tmp_zeros=numpy.zeros([video_extend])    
        score_action=numpy.concatenate((tmp_zeros,score_action,tmp_zeros))
        tmp_cell = video_gap
        tmp_x = [-tmp_cell/2-(video_extend-1-ii)*tmp_cell for ii in range(video_extend)] + \
                 [tmp_cell/2+ii*tmp_cell for ii in range(video_scale)] + \
                  [tmp_cell/2+seg_xmaxs[-1] +ii*tmp_cell for ii in range(video_extend)]
        f_action=interp1d(tmp_x,score_action,axis=0)
        feature_bsp=[]
    
        for idx in range(len(pdf)):
            xmin=pdf.xmin.values[idx]
            xmax=pdf.xmax.values[idx]
            xlen=xmax-xmin
            xmin_0=xmin-xlen * opt["bsp_boundary_ratio"]
            xmin_1=xmin+xlen * opt["bsp_boundary_ratio"]
            xmax_0=xmax-xlen * opt["bsp_boundary_ratio"]
            xmax_1=xmax+xlen * opt["bsp_boundary_ratio"]
            #start
            plen_start= (xmin_1-xmin_0)/(num_sample_start-1)
            plen_sample = plen_start / num_sample_interpld
            tmp_x_new = [ xmin_0 - plen_start/2 + plen_sample * ii for ii in range(num_sample_start*num_sample_interpld +1 )] 
            tmp_y_new_start_action=f_action(tmp_x_new)
            tmp_y_new_start = [numpy.mean(tmp_y_new_start_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_start) ]
            #end
            plen_end= (xmax_1-xmax_0)/(num_sample_end-1)
            plen_sample = plen_end / num_sample_interpld
            tmp_x_new = [ xmax_0 - plen_end/2 + plen_sample * ii for ii in range(num_sample_end*num_sample_interpld +1 )] 
            tmp_y_new_end_action=f_action(tmp_x_new)
            tmp_y_new_end = [numpy.mean(tmp_y_new_end_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_end) ]
            #action
            plen_action= (xmax-xmin)/(num_sample_action-1)
            plen_sample = plen_action / num_sample_interpld
            tmp_x_new = [ xmin - plen_action/2 + plen_sample * ii for ii in range(num_sample_action*num_sample_interpld +1 )] 
            tmp_y_new_action=f_action(tmp_x_new)
            tmp_y_new_action = [numpy.mean(tmp_y_new_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_action) ]
            tmp_feature = numpy.concatenate([tmp_y_new_action,tmp_y_new_start,tmp_y_new_end])
            feature_bsp.append(tmp_feature)
        feature_bsp = numpy.array(feature_bsp)
        numpy.save("./output/PGM_feature/"+video_name,feature_bsp)


def PGM_proposal_generation(opt):
    video_dict= load_json(opt["video_anno"])
    video_list=video_dict.keys()#[:199]
    num_videos = len(video_list)
    num_videos_per_thread = num_videos/opt["pgm_thread"]
    processes = []
    for tid in range(opt["pgm_thread"]-1):
        tmp_video_list = video_list[tid*num_videos_per_thread:(tid+1)*num_videos_per_thread]
        p = mp.Process(target = generateProposals,args =(opt,tmp_video_list,video_dict,))
        p.start()
        processes.append(p)
    
    tmp_video_list = video_list[(opt["pgm_thread"]-1)*num_videos_per_thread:]
    p = mp.Process(target = generateProposals,args =(opt,tmp_video_list,video_dict,))
    p.start()
    processes.append(p)
    
    for p in processes:
        p.join()

def PGM_feature_generation(opt):
    video_dict=getDatasetDict(opt)
    video_list=video_dict.keys()
    num_videos = len(video_list)
    num_videos_per_thread = num_videos/opt["pgm_thread"]
    processes = []
    for tid in range(opt["pgm_thread"]-1):
        tmp_video_list = video_list[tid*num_videos_per_thread:(tid+1)*num_videos_per_thread]
        p = mp.Process(target = generateFeature,args =(opt,tmp_video_list,video_dict,))
        p.start()
        processes.append(p)
    
    tmp_video_list = video_list[(opt["pgm_thread"]-1)*num_videos_per_thread:]
    p = mp.Process(target = generateFeature,args =(opt,tmp_video_list,video_dict,))
    p.start()
    processes.append(p)
    
    for p in processes:
        p.join()

