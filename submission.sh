NAME=ZHOU_SHAOWEN
USER_ID=EXXXXXXX
FOLDER_NAME=${NAME}_${USER_ID}

rm -r submission

mkdir submission
mkdir submission/$FOLDER_NAME
cp -r Test_data submission/$FOLDER_NAME
cp {Clustering.py,DecisionTreeRegressor.py,GradientBoostingRegressor.py,readme.txt,requirements.txt,Assignment.pdf} submission/$FOLDER_NAME
cd submission
zip -r $FOLDER_NAME.zip $FOLDER_NAME
