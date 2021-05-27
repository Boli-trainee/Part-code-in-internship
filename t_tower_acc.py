#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import paddlex as pdx
import json
import os
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool

import time
import pandas as pd
import pymysql
import requests

"""
db = pymysql.connect(
    host="192.168.173.2",
    port=3306,
    user="root",
    password="root123",
    database="ephoto",
    charset="utf8mb4",
    connect_timeout=31536000,
)
"""

db = pymysql.connect(
    host="192.168.1.86",
    port=33306,
    user="root",
    password="root_mysql8",
    database="ephoto",
    charset="utf8mb4",
    connect_timeout=31536000,
)


def query(sql, commit=False):
    cursor = db.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    cursor.close()
    if commit:
        db.commit()
    return data


df = pd.read_sql(
    "SELECT photo.path,photo.name,photo.code,`photo_album`.name AS album,`tower_type`.name AS type ,`tower_type`.code AS t_code FROM photo  LEFT JOIN `photo_album` ON photo.`album_id`=`photo_album`.id LEFT JOIN `tower` ON `photo_album`.`target_id`=tower.id LEFT JOIN `tower_type` ON tower.`type_code`=`tower_type`.code LEFT JOIN `tower_part_tag` ON photo.id=`tower_part_tag`.`photo_id` WHERE album_id IN (SELECT `album_id`  FROM `task_album` WHERE `status`=2) AND photo.quality=0 AND `tower_part_tag`.`user_id`>1;",
    db,
)
df_use = df.drop_duplicates()
df_useful = df_use.loc[df["code"].str.len() >= 5]


################################ 需要改动的地方起点 #######################################

# 塔型和数字的对应关系：
# 1: T字塔
# 2: 交流同塔并架直线塔
# 3: 交流同塔并架耐张塔
# 4: 交流干字耐张塔
# 5: 干字直线塔
# 6: 拉V塔
# 7: 猫头塔
# 8: 直流干字耐张塔
# 9: 直流紧凑耐张塔
# 10: 酒杯塔
# 11: 门型塔
# 将要预测的塔型的对应数字复制给part_code
part_count = 1

# 映射数据，换塔型和映射关系时需要更新映射关系。注意要按照左右进行分开！
# keys为模型返回值
keys = ["01", "02", "03", "0106", "0107", "0108", "0109", "0110", "0111", "0112", "0113", "0114",
              "0115", "0124", "0125", "0126", "0127", "0130", "0131"]

# values为keys对应的部件编码（真值），需要按照左右进行区分。
values_left = [1, 2, 3, 6, 7, 8, 9, 10, 10, 12,
               13, [14, 28], [15, 29], 24, 24, 26, 27, 11, 25]
values_right = [1, 2, 3, 8, 9, 6, 7, 24, 25, [26, 27],
                26, [14, 28], [15, 29], 10, 11, [12, 13], 12, 24, 10]

# values表示模型返回值对应的真值标签，比如“01”对应1。（数据库中的真值直接采用数字）
# 此处partkeys的值需要和keys一一对应写入，用于使后面的输出更加直观
partkeys = ["塔头", "基础", "号牌", "近地线挂点", "近地线防振锤", "远地线挂点", "远地线防振锤", "近垂直串塔端挂点", "近V串近塔端挂点", "近绝缘子串/近相近V串", "近相远V单串",
            "导线端挂点", "导线防振锤", "远垂直串塔端挂点", "远V串近塔端挂点", "远绝缘子串/远相近V串", "远相远V单串", "近V串远塔端挂点", "远V串远塔端挂点"]
# 数据库的路径，若数据库位置不变则不需要变动
path_origin = "/media/hangpai/tool/model/"


# 将path_to_model替换为相应的模型路径
path_to_model = "/media/hangpai/libo/models/t_tower/t_tower_model_519/inference_model/"

# json文件读取左右侧信息部分，修改json文件的路径
jsonPath = "/media/hangpai/libo_training/pos.json"

################################ 需要改动的地方终点 #######################################


print("Loading model...")
model = pdx.deploy.Predictor(path_to_model, use_gpu=False)
print("Model loaded.")


assert os.path.isfile(jsonPath)
with open(jsonPath) as inf:
    Dict = json.load(inf)
    # 根据Dict里面的两个信息来提取左右侧信息：线路->塔编号

# imgpath表示待预测的图片路径


def predict(imgpath):

    im = cv2.imread(imgpath)
    im = im.astype('float32')

    result = model.predict(im)

    # 将图片送入模型，返回模型给出的类型
    return result[0]["category"]


list_value = []
for row_id, row in df_useful.iterrows():
    tmp = row["path"].split('/')[1]
    # print(tmp)
    # break
    list_value.append(tmp)

df_useful["sort_value"] = list_value


zero_values = [0] * len(values_left)
# time_count = 0
for tower_type_id, tower_type in df_useful.groupby("type"):
    tower_excel = pd.DataFrame()
    line_excel = pd.DataFrame()
    for line_id, line in tower_type.groupby("sort_value"):  # 进入不同的线路
        # time_count += 1
        # if time_count != part_count:
        #     continue
        count_line = 0
        count_line_total = 0
        count_tower = 0
        for album_id, album in line.groupby("album"):
            # 进入某个具体塔下面
            # print(album_id, "   ", album)
            TaPartdict_left = dict(zip(keys, values_left))  # 用于映射
            TaPartdict_right = dict(zip(keys, values_right))  # 用于映射
            CandidateDict = dict(zip(keys, zero_values))  # 用于计数
            img_list = album.values.tolist()
            count_album = 0
            count_album_total = 0
            count_tower += 1
            for img in img_list:
                # if img[2][2:4] != "00":
                #     flag = 1
                #     break
                code = int(img[2][:2])  # 真值标签
                result = str(predict(os.path.join(
                    path_origin, img[0], img[1])))  # 预测值
                # print(result,"  ", code)
                pos, line = img[0].split('/')[1], img[0].split('/')[2]
                # 考虑字典中没有对应键的情况
                try:
                    Dict[pos][line]
                except:
                    continue
                if Dict[pos][line] == "右侧":
                    TaPartdict = TaPartdict_right
                else:
                    TaPartdict = TaPartdict_left
                count_album_total += 1
                # print(code, "  ", TaPartdict[result])
                if code == TaPartdict[result]:
                    count_album += 1
                elif type(code) != type(TaPartdict[result]):
                    if code in TaPartdict[result]:
                        count_album += 1
                # else:
                #     CandidateDict[result] += 1
            if not count_album_total:
                continue
            accuracy = count_album / count_album_total
            count_line += count_album
            count_line_total += count_album_total
            # TaPartValues = CandidateDict.values()
            # TaPartdict = dict(zip(partkeys, TaPartValues))
            # SortTa = Counter(TaPartdict).most_common(6)

            print("线路：{}，塔号：{}，图片数量：{}，塔识别准确率{:.4f}\n".format(
                line_id, album_id, count_album_total, accuracy))
            tmp = {"线路": line_id, "塔号": album_id,
                   "图片数量": count_album_total, "塔识别准确率": accuracy}
            tower_excel = tower_excel.append(tmp, ignore_index=True)
            # with open("tower_record.txt", "a") as f:
            #     f.write("线路：{}，塔号：{}，塔数量：{}，塔识别准确率{:.4f}\n".format(
            #         line_id, album_id, count_album_total, accuracy))

            # for i in range(len(values)-3):
            #     print("{}的第{}个候选项为:{},数量为：{}\n".format(
            #         part_code[0], i+1, ThreeTa[i][0], ThreeTa[i][1]))
            #     with open("record.txt", "a") as f:
            #         f.write("{}的第{}个候选项为:{},数量为：{}\n".format(
            #             part_code[0], i+1, ThreeTa[i][0], ThreeTa[i][1]))

            # with open("tower_record.txt", "a") as f:
            #     f.write("\n")
        if not count_album_total:
            continue
        line_accuracy = count_line / count_line_total
        # print("*" * 20)
        # print("线路：{}，塔数量：{}， 图片数量：{}，线路识别准确率为{:.4f}\n".format(
        #     line_id, count_tower, count_line_total, line_accuracy))
        # print("*" * 20)
        tmp = {"线路": line_id, "塔数量": count_tower,
               "图片数量": count_line_total, "线路识别准确率": line_accuracy}
        line_excel = line_excel.append(tmp, ignore_index=True)
        print(line_excel)

    tower_excel.to_excel(
        "/media/hangpai/libo/models/t_tower/t_tower_report_519/t_tower_tower_record.xlsx")
    line_excel.to_excel(
        "/media/hangpai/libo/models/t_tower/t_tower_report_519/t_tower_line_record.xlsx")
    break
    # print("线路：{}，塔数量：{}，线路识别准确率为{:.4f}\n".format(
    #     line_id, count_line_total, line_accuracy))
    # print("*" * 20)
    # with open("line_record.txt", "a") as f:
    #     f.write("线路：{}，塔数量：{}，线路识别准确率为{:.4f}\n".format(
    #         line_id, count_line_total, line_accuracy))
    break
