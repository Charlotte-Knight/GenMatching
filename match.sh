#!/usr/bin/env bash

cd /vols/cms/mdk16/ggtt/DY_Skims
source env/bin/activate
python match.py $1
