clear;
clc;
close all;
road_point_index = 4;

if road_point_index == 1
   road_point = 'fog_1508_1';
   image_number = 150;
   d1 = 750;
   d2 = 541;
   v1 = 817;
   v2 = 953;
elseif road_point_index == 2
   road_point = 'fog_1508_2';
   image_number = 90;
   d1 = 750;
   d2 = 541;
   v1 = 818;
   v2 = 953;
elseif road_point_index == 3
   road_point = 'fog_1508_3';
   image_number = 120;
   d1 = 1181.8;
   d2 = 454.8;
   v1 = 743;
   v2 = 941;
elseif road_point_index == 4
   road_point = 'fog_1508_4';
   image_number = 151;
   d1 = 2461.11;
   d2 = 360;
   v1 = 647;
   v2 = 950;
elseif road_point_index == 5
   road_point = 'fog_1508_5';
   image_number = 150;
   d1 = 1037.25;
   d2 = 659.375;
   v1 = 773;
   v2 = 921;
elseif road_point_index == 6
   road_point = 'fog_1508_6';
   image_number = 120;
   d1 = 1181.8;
   d2 = 454.8;
   v1 = 753;
   v2 = 936;
end 
Vh = 540 - 0;
AA = (d1-d2)/(1/(v1-Vh)-1/(v2-Vh));

filepath='.\total_median_mean';
data_save_address_name=strcat(filepath,'\',road_point,'_','total_median_mean','.mat');
load(data_save_address_name);
t=eval(strcat(road_point,'_','total_median_mean'));
for i=1:length(t)
    d = AA/(t(i)-Vh);
    dd(i,1)=d;
end

eval(strcat(road_point,'_','after_address','=dd;'));
save_address_name=strcat('.\after_address\',road_point,'_','after_address','.mat');
save(save_address_name,strcat(road_point,'_','after_address'))

