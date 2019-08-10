clear;
clc;
close all;


subject_No=1; 
tester_name='zyx_16';
road_point_index = 6;  
if road_point_index == 1
   road_point = 'fog_1508_1';
   image_number = 150;   
elseif road_point_index == 2
   road_point = 'fog_1508_2';
   image_number = 90;   
elseif road_point_index == 3
   road_point = 'fog_1508_3';
   image_number = 120;   
elseif road_point_index == 4
   road_point = 'fog_1508_4';
   image_number = 151;   
elseif road_point_index == 5
   road_point = 'fog_1508_5';
   image_number = 150;   
elseif road_point_index == 6
   road_point = 'fog_1508_6';
   image_number = 120;   
end 

filepath='.\zyx_16';
for j=1:5
    %data_save_address_name=strcat(filepath,'\points_details\fog_1508_',num2str(road_point_index),'\Vv_',road_point,'_','subject_',num2str(subject_No),'_',num2str(j),'.mat');
    data_save_address_name=strcat(filepath,'\Result_data\points_details\fog_1508_',num2str(road_point_index),'\Vv_',road_point,'_','subject_',num2str(subject_No),'_',num2str(j),'.mat');
    load(data_save_address_name);
    t=eval(strcat('Vv_',road_point,'_','subject_',num2str(subject_No),'_',num2str(j),';'));
    m(:,j)=t(:,2);
end
for i=1:length(m)
    %n(i,1)=mean(m(i,:));
    n(i,1)=median(m(i,:));
end
eval(strcat(road_point,'_',tester_name,'_median','=n;'));
save_address_name=strcat('.\median\',road_point,'\',road_point,'_',tester_name,'_median','.mat');
save(save_address_name,strcat(road_point,'_',tester_name,'_median'))   
