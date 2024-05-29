from basic_utils.Evaluation import calculate_metrics, calc_scores
from basic_utils.Processing import get_embeddings_batch
from basic_utils.FileOP import traversal_dir, images_process, SavetoYaml

import yaml
import os
import time
import sys
import torch

baseline = ["TSconesP", "Reference", "ProSpect", "DreamBooth", "SVDiff", "TI", "XTI", "TSconesP_Q", "Cones"]
CUDA_VISIBLE_DEVICES = int(sys.argv[2])
choice = int(sys.argv[1])
timestamp = sys.argv[3] # '24-0213-20' # time_set()

dir1 = "datasets/"

my_dict1 = {
            "clock": "clock",
            "elephant": "sculpture",
            "mug_skulls":"ceramic",
            "physics_mug":"cup",
            "red_teapot":"teapot",
            "round_bird":"sculpture",
            "thin_bird":"sculpture"
            }

dir2 = "dreambooth/dataset/"

my_dict2 = {
"backpack":"backpack",
"backpack_dog":"backpack",
"bear_plushie":"stuffed_animal",
"berry_bowl":"bowl",
"can":"can",
"candle":"candle",

"cat":"cat",
"cat2":"cat",
"clock":"clock",

"duck_toy":"toy",
"fancy_boot":"boot",
"grey_sloth_plushie":"stuffed_animal",
"monster_toy":"toy",
"pink_sunglasses":"glasses",
"poop_emoji":"toy",
"rc_car":"toy",
"red_cartoon":"cartoon",
"robot_toy":"toy",
"shiny_sneaker":"sneaker",
"teapot":"teapot",
"vase":"vase",
"wolf_plushie":"stuffed_animal",
"gal_gadot":"beauty"
}

checkpoint_num =  [300,600,900]


def eval_data( my_dict1, path, checkpoint_num, source, metrics, save):
    
    for key, value in my_dict1.items():
        for cn in checkpoint_num:
            # print((path, source+f'{key}', key, checkpoint_num))
            _, _, _, metric = SGE(path, source+f'{key}', key, cn, value, save=save)
            metrics[cn].append(metric)            
            # time.sleep(2)
            # exit()

    return

def eval_datas( my_dict1, my_dict2, path, source, dir1, dir2, checkpoint_num, save=True):
    metrics = { cn: [] for cn in checkpoint_num}
    eval_data( my_dict1, path+dir1, checkpoint_num, source+dir1, metrics, save)
    eval_data( my_dict2, path+dir2, checkpoint_num, source+dir2, metrics, save)
    avgs_m = { cn: {} for cn in checkpoint_num}
    for cn in checkpoint_num:
        avg_m = average_metrics(metrics[cn])
        avgs_m[cn]['txt_avg_metric'] = avg_m[1]
        avgs_m[cn]['img_avg_metric'] = avg_m[0]
    SavetoYaml(path+"avgs_m.yaml", avgs_m)
    print(avgs_m)
    return

def save2file(img_scores, txt_scores, emb_dict, save_dir):
    avg, min_val, max_val, neg_error, pos_error = calculate_metrics(img_scores)
    img_metric = {
        "Average": avg,
        "Min": min_val,
        "Max": max_val,
        "Negative Error": neg_error,
        "Positive Error": pos_error
    }
    
    avg, min_val, max_val, neg_error, pos_error = calculate_metrics(txt_scores)
    txt_metric = {
        "Average": avg,
        "Min": min_val,
        "Max": max_val,
        "Negative Error": neg_error,
        "Positive Error": pos_error
    }
    scores ={"txt_scores":txt_scores,
             "img_scores":img_scores}
    metric ={"txt_metric":txt_metric,
             "img_metric":img_metric}
    # print(metric)

    torch.save( emb_dict, save_dir + "/emb_dict.pt")
    SavetoYaml(save_dir + "/metric.yaml", metric)
    SavetoYaml(save_dir + "/scores.yaml", scores)
    return img_scores, txt_scores, emb_dict, metric

import torch

def average_metrics( metrics):

    total_img_metrics = {
        "Average": 0,
        "Min": 0,
        "Max": 0,
        "Negative Error": 0,
        "Positive Error": 0
    }

    total_txt_metrics = {
        "Average": 0,
        "Min": 0,
        "Max": 0,
        "Negative Error": 0,
        "Positive Error": 0
    }


    total_count = 0
    for metric in metrics:
        # print(metric)
        for key, value in metric["img_metric"].items():
            total_img_metrics[key] += value
        for key, value in metric["txt_metric"].items():
            total_txt_metrics[key] += value
        total_count += 1

    avg_img_metrics = {}
    avg_txt_metrics = {}
    for key, value in total_img_metrics.items():
        avg_img_metrics[key] = value / total_count
    for key, value in total_txt_metrics.items():
        avg_txt_metrics[key] = value / total_count

    return avg_img_metrics, avg_txt_metrics


def load4file(save_dir):

    emb_dict = torch.load(save_dir + "/emb_dict.pt")


    with open(save_dir + "/metric.yaml", "r") as file:
        metric = yaml.safe_load(file)


    with open(save_dir + "/scores.yaml", "r") as file:
        scores = yaml.safe_load(file)


    txt_scores = scores["txt_scores"]
    img_scores = scores["img_scores"]

    return img_scores, txt_scores, emb_dict, metric


def SGE( path, source, class_def, cn, instance, save=True):
    if save:
        image_dirs = traversal_dir( path, class_def, str(cn)+'/samples')[:60]
        source_dirs = traversal_dir( source, '', '')
        
        source_batch = images_process(source_dirs) 
        image_batch = images_process(image_dirs) 

        embeddings_batch = get_embeddings_batch(image_batch)
        # print(embeddings_batch[0].shape) # [1, 768]
        # print(len(embeddings_batch)) # 60
        source_txt = "a photo of a " + instance
        print(source_txt)
        source_embeddings_img = get_embeddings_batch(source_batch, avg=True)
        source_embeddings_txt = get_embeddings_batch(source_txt, avg=True, img=False)
        # print(source_embeddings.shape) # [1, 768]
        img_scores = calc_scores( source_embeddings_img, embeddings_batch)
        txt_scores = calc_scores( source_embeddings_txt, embeddings_batch)

        # print(scores)
        
        save_dir = os.path.join( path, class_def, str(cn))
        emb_dict = {"source_embeddings_img": source_embeddings_img,
                    "source_embeddings_txt": source_embeddings_txt,
                    "embeddings_batch": embeddings_batch }
        
        return save2file(img_scores, txt_scores, emb_dict, save_dir)
        
    else:
        load_dir = os.path.join( path, class_def, str(cn))
        return load4file(load_dir)
    
    

if __name__ == "__main__":
    ex_name = baseline[choice] 
    run_name = ex_name + '_'+ timestamp
    path = f"/media/sdata/chy/logs/evaluation/{run_name}/generated/"
    source = f"/home/hyc/newcones/custom-diffusion/data/"
    print(path)
    if ex_name in [ "TSconesP_Q", "DreamBooth",  "SVDiff", "TI", "Cones"]:
        # checkpoint_num = [ '/checkpoint-' + str(i) for i in checkpoint_num]
        eval_datas( my_dict1, my_dict2, path, source, dir1, dir2, checkpoint_num)
        # NOTE(beixue): save = False 平均处理后的数据, save = True 表明处理数据
        eval_datas( my_dict1, my_dict2, path, source, dir1, dir2, checkpoint_num, save=False)
    elif ex_name == "TSconesP":
        checkpoint_num = [(i+1)*300 for i in range(5)]
        eval_datas( my_dict1, my_dict2, path, source, dir1, dir2, checkpoint_num)
        # NOTE(beixue): save = False 平均处理后的数据, save = True 表明处理数据
        eval_datas( my_dict1, my_dict2, path, source, dir1, dir2, checkpoint_num, save=False)
    elif baseline[choice] == "XTI":

        checkpoint_num = [150,300,450]
        # checkpoint_num = [ '/checkpoint-' + str(i) for i in checkpoint_num]
        eval_datas( my_dict1, my_dict2, path, source, dir1, dir2, checkpoint_num)
        eval_datas( my_dict1, my_dict2, path, source, dir1, dir2, checkpoint_num, save=False)
    elif baseline[choice] == "ProSpect":
        checkpoint_num = ['best']
        # checkpoint_num = [ '/checkpoint-' + str(i) for i in checkpoint_num]
        eval_datas( my_dict1, my_dict2, path, source, dir1, dir2, checkpoint_num)
    else:
        print("notfoundchoice error!")