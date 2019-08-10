clear;
clc;
close all;
tester_totalnumber = 16;
road_point_index = 1;  
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

    

filepath='.\median';
for j=1:tester_totalnumber
    if j == 1
        tester_name = 'cm';
    elseif j ==2
        tester_name = 'cx';
    elseif j ==3
        tester_name = 'dxx';
    elseif j ==4
        tester_name = 'fs';
    elseif j ==5
        tester_name = 'gx';
    elseif j ==6
        tester_name = 'gyx';
    elseif j ==7
        tester_name = 'ldz';
    elseif j ==8
        tester_name = 'lhj';
    elseif j ==9
        tester_name = 'qjp';
    elseif j ==10
        tester_name = 'swj';
    elseif j ==11
        tester_name = 'whw';
    elseif j ==12
        tester_name = 'wq';
    elseif j ==13
        tester_name = 'wt';
    elseif j ==14
        tester_name = 'xxb';
    elseif j ==15
        tester_name = 'yzc';
    elseif j ==16
        tester_name = 'zyx';
    end

    data_save_address_name=strcat(filepath,'\',road_point,'\',road_point,'_',tester_name,'_',num2str(j),'_','median','.mat');
    load(data_save_address_name);
    t=eval(strcat(road_point,'_',tester_name,'_',num2str(j),'_','median'));
    m(:,j)=t(:,1);
end
for i=1:length(m)
    n(i,1)=mean(m(i,:));
end
eval(strcat(road_point,'_','total_median_mean','=n;'));
save_address_name=strcat('.\total_median_mean\',road_point,'_','total_median_mean','.mat');
save(save_address_name,strcat(road_point,'_','total_median_mean'))
