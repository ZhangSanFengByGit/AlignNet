dirs = dir('G:\RGB-T_unfit_dataset');

file_num = 108;

for i = 112:size(dirs)
    
    if ~exist(['G:\data\'  num2str(file_num)],'dir')
          mkdir(['G:\data\'  num2str(file_num)]);
    end
    
    try
        current_dir = dirs(i).name;
        rgb_v=VideoReader(['G:\RGB-T_unfit_dataset\'  current_dir   '\i.avi']);
        t_v=VideoReader(['G:\RGB-T_unfit_dataset\'  current_dir  '\v.avi']);
    catch Errorinfo
        disp(current_dir);
        continue;
    end
    
    rgb_num=rgb_v.NumberOfFrames;
    t_num=t_v.NumberOfFrames;
    
    if ~exist(['G:\data\', num2str(file_num) ,'/rgb'],'dir')
          mkdir(['G:\data\', num2str(file_num) ,'/rgb']);
    end
    
    for j=1:rgb_num
        if j>1000
            break
        end
        image_name=strcat('G:\data\', num2str(file_num) ,'/rgb/',num2str(j),'.jpg');   
        frame=read(rgb_v,j);           
        imwrite(frame,image_name,'jpg');
        
    end
    
    if ~exist(['G:\data\', num2str(file_num) ,'/t/'],'dir')
          mkdir(['G:\data\', num2str(file_num) ,'/t/']);
    end
    
    for j=1:t_num
        if j>1000
            break
        end
        image_name=strcat('G:\data\', num2str(file_num) ,'/t/',num2str(j),'.jpg');   
        frame=read(t_v,j);           
        imwrite(frame,image_name,'jpg');
        
    end
    file_num = file_num + 1;
end