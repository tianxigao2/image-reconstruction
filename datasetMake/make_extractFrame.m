close all
clear all
clc

cmp30 = Extractor('compressedSample_30.avi');
csvwrite('cmp30.csv', cmp30);

cmp50 = Extractor('compressedSample_50.avi');
csvwrite('cmp50.csv', cmp50);

cmp75 = Extractor('compressedSample_75.avi');
csvwrite('cmp75.csv', cmp75);

uncmp = Extractor('uncompressedSample.avi');
csvwrite('uncmp.csv', uncmp);    
