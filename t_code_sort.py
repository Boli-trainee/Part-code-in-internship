#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
import pymysql
import requests
# from predict import predict

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

target_path = "/media/hangpai/libo/" 
origin_path = "/media/hangpai/tool/model/"

for tower_type in df.groupby("type"):
    print(tower_type[0])
    path = target_path + str(tower_type[0])
    if not os.path.exists(path):
        os.mkdir(path)
    Talist = tower_type[1].values.tolist()
    for Ta in Talist:
        print(Ta[0]+Ta[1])
        if Ta[2]:
            code = Ta[2][:2]
            Tapath = path +'/'+ str(code)
            if not os.path.exists(Tapath):
                os.mkdir(Tapath)
            shutil.copy(origin_path + Ta[0] + '/' + Ta[1], Tapath + '/' +  Ta[1])
