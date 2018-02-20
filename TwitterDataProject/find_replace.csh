#!/usr/bin/env bash

FIND='scree_ame'
REPLACE='screen_name'
for file in *.log
do
    sed -i -e "s/${FIND}/${REPLACE}/g" ${file}
done
