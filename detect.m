clear
img=imread('C:\Users\lenovo\Desktop\img\1.BMP');
img=img.*uint8(img<100);
% img=rgb2gray(img);
img=histeq(img);
subplot(3,2,1)
imshow(img)
fft=fft2(img);
sfft=fftshift(fft);
r=real(sfft);
i=imag(sfft);
% A=sqrt(r.^2+i.^2);
% A=(A-min(min(A)))./(max(max(A))-min(min(A)))*255;
% rA=real(A);

subplot(3,2,2)
imshow(r)

[N1,N2]=size(r);
d0=100; 
%d0是终止频率
n1=fix(N1/2);
n2=fix(N2/2);
%n1，n2指中心点的坐标，fix（）函数是往 0  取整
for i=1:N1
  for j=1:N2
%     d=sqrt((i-n1)^2+(j-n2)^2);
%     h=exp(-d*d/(2*d0*d0));
%     result(i,j)=h*r(i,j);
    d=sqrt((i-n1)^2+(j-n2)^2);   %理想低通滤波，求距离  
    if d<=d0  
        h(i,j)=1;  
    else  
        h(i,j)=0;  
    end  
    result(i,j)=h(i,j)*r(i,j);    
  end
end

result=ifftshift(result);
X2=ifft2(result);
%傅里叶去中心化以及反变换
final=uint8(real(X2));
final=histeq(final);
subplot(3,2,3)
imshow(final);
his=histeq(final);
subplot(3,2,4);
imshow(his);

edg=edge(his,'sobel');
subplot(3,2,5);
imshow(edg);

% x=[];y=[];z=[];
% [a,b]=size(img);
% for i=1:a
%     for j=1:b
%         x(end+1)=i;
%         y(end+1)=j;
%         z(end+1)=img(i,j);
%     end
% end
% plot3(x,y,z)
% point_detect_number=0;
% line_detect_number=0;
% block_detect_number=0;
% 
% detect_noise=imread('C:\Users\lenovo\Desktop\img\IMG_1725.BMP');
% 
% detect_gabor=0;
% for f0=[1/4,1/8,1/16,1/32]
%     for theta=[0,pi/4,pi/2,pi*3/4]
%         x=0;
%         for i=linspace(-1,1,7)
%             x=x+1;
%             y=0;
%             for j=linspace(-1,1,7)
%                 y=y+1;
%                 z(y,x)=compute(i,j,f0,theta);
%             end
%         end
%         filtered=filter2(z,detect_noise,'valid');
%         f=abs(filtered);
%         detect_gabor=detect_gabor+f;
%     end
% end
% detect_gabor=detect_gabor/max(detect_gabor(:));
% detect_med_filter=medfilt2(detect_noise,[3,3]);
% 
% sx=fspecial('sobel');
% sy=sx';
% gx=imfilter(detect_gabor,sx,'replicate');
% gy=imfilter(detect_gabor,sy,'replicate');
% grad=sqrt(gx.*gx+gy.*gy);
% grad=grad/max(grad(:));
% h=imhist(grad);
% hp(1)=0;
% bar(hp,0);
% T=otsuthresh(hp);
% T*(numel(hp)-1);
% g=imbinarize(detect_gabor,T);
% g1=~g;
% g2=edge(g1,'sobel');
% 
% if(max(g2(:)))
%     se90=strel('line',5,90);
%     se0=strel('line',5,0);
%     BW2=imdilate(g2,[se90,se0]);
%     BW3=BW2|g1;
%     [L,m]=bwlabel(BW3,8);
%     status=regionprops(L,'all');
%     imshow(g);
%     hold on
%     point_area_max=250;
%     line_area_max=10000;
%     for i=1:m
%         if(status(i).Area<point_area_max)
%             point_defect_number=point_defect_number+1;
%         else
% %             if(status(i).Area<line_area_max&&status(i).MahorAxis)
%             point_defect_number=point_defect_number+1;
%         end
%         rectangle('position',status(i).BoundingBox,'Curvature',[1,1],'edgecolor','r');
%     end
% end
%    
% 
%         