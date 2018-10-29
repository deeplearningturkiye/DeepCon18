clc;
clear all;

start = 1;
finish = 322;
sayac = 1;


% ON ISLEMELI START

for i=start:finish
    
        if (i<10)
            path = sprintf('all-mias/mdb00%d.pgm',i);
        elseif(i<100)
            path= sprintf('all-mias/mdb0%d.pgm',i);
        else
            if (i==144)     % etiketi 2 sýnýf için de geçerli oldugundan bu resim alýnmadý
                continue;
            end
             path = sprintf('all-mias/mdb%d.pgm',i);
        end
        
        
        resim = imread(path);
        ResimOrijinal=im2double(resim); 
        
        Resim_bw=im2bw(ResimOrijinal,0.2);
      
           
        labeled_I=bwlabel(Resim_bw); 
        
     
        uniq=unique(labeled_I(:)); % kac tane uniq connected component buldu, kac tane etiket kullandim

        count=hist(labeled_I(:),uniq); % uniq deðerlerin bütün matristeki historamýný bulur. 
                                       %kac tane uniq var, her bir etiketten kac tane deger var

        A=[uniq count'];  % her bir etiketten kac deger var , matrisler birlestirildi.

        new=zeros(1024,1024); % 1024 * 1024 'e sifir matris olusturuldu. amac max. connected componentleri buraya aktarmak.

           
        value=max(A(2 :end,2));   % max connected komponentleri bul. 0 lar siyah oldugundan 2.satir dan baslayarak 2. sutunu tara

        [row column]=find(A==value); % A matrisindeki max. komponentin satir ve sutun bilgisini bul
        
        row=row-1;

        new(find(labeled_I==row))=1;  % labeled resimdeki max degerlikli bolgenin koordinatlar?n? bul new matrisinde oraya 1 leri yaz.

      
        mult= (new).*ResimOrijinal;  % ana matris ile new matrisini carp istenilen bolgenin ilk halini getir.

%          figure;
%          imshow(mult);
        
        
        
        % Feature Extraction

           
%               lbpFeatures256x256(sayac,:) = extractLBPFeatures(mult,'CellSize',[256 256],'Normalization','None'); 
%               hogFeatures256x256(sayac,:) = extractHOGFeatures(mult,'CellSize',[256 256]);   
                lbpFeaturesFull(sayac,:) = extractLBPFeatures(mult);
%               hogFeaturesFull(sayac,:) = extractHOGFeatures(mult);
%               lbpFeatures32x32(sayac,:) = extractLBPFeatures(mult,'CellSize',[32 32],'Normalization','None');
%               hogFeatures32x32(sayac,:) = extractHOGFeatures(mult,'CellSize',[32 32]);
% 
%               sonuc=GLCMFeatures(mult);
%               A = struct2cell(sonuc);
%               V01_3_GLCMFeatures(sayac,:) = cat(2,A{:});

% 
          sayac = sayac + 1;
end

filename = 'lablenumeric.xlsx';
label = xlsread(filename);

%ARFF Formati
mat2arff(lbpFeaturesFull,label ,'data.arff','a','0,1');



%EXCEL Formati 
% filename = 'testdata.xlsx';% 
% xlswrite(filename,hogFeatures256x256)


%CSV Formati 
% csvwrite('LBP_Features_256X256.csv', lbpFeatures256x256);
% csvwrite('HOG_Features_256x256.csv', hogFeatures256x256);




        