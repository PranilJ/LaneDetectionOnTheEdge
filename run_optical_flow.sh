#!/bin/bash 

#arg1='i1.jpg'
#python3 ./opticalflow.py arg1 i2.jpg b1.jpg b2.jpg

cd results/0601/
for SUBFOLDER in *; 
do 
echo $SUBFOLDER; 
for FILE in $SUBFOLDER/*;
do
echo $FILE;
cd ..;
done
done	
 
