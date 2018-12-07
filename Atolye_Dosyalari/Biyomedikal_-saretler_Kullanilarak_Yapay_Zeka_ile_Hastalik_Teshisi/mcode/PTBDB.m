        clear;
        clc;
        %fid = fopen('ptbdb-Myocardial_infarction.txt'); 
       fid = fopen('ptbdb_control.txt'); 
        dates = textscan(fid,'%s');
        dosyayeri='ptbdb/';
        fclose(fid);
        %[signal,Fs,tm]=rdsamp('ptbdb/patient001/s0010_re',[],10000);
        %plot(tm,signal(:,1))
        records=dates{1,1};
        for i=1:1 %size(records,1)
                i
                P(i,1) =string(records(i,1));
                filename=char(dosyayeri+P(i,1));
                [signal,Fs,tm]=rdsamp(filename,[],[]);
                ecg_lead=signal(:,1);
                
                %%%Eliminate Baseline Drift
                s1=ecg_lead;s2=smooth(s1,150);ecg_smooth=s1-s2;
                
                %%%
                filtered_signal=ecg_smooth;
                n=30;
                for j=n:length(ecg_smooth)
                    temp=0;
                    for k=j-(n-1):j
                        temp=temp+ecg_smooth(k);
                        mean_signal=temp/n;
                    end
                    filtered_signal(j)=mean_signal;       
                end
          
                % 60Hz sebeke gurultusunu ve harmoniklerini bastiran filtre
                B=conv([1 1],[0.6310 -0.2149 0.1512 -0.1288 0.1227 -0.1288 0.1512 -0.2149 0.6310]);
                A=1;
                freqz(B,A);
                comb=filter(B,A,filtered_signal);
                
                % Differentiator
                differentiatorcikisi=diff(comb);
 
                % Squaring Operation
                kare=differentiatorcikisi.*differentiatorcikisi;
 
                % Moving Window Integrator
                window=ones(1,30); % N=30 moving window
                integral= medfilt1(filter(window,1,kare),10);
                delay = ceil(length(window)/2);
                integral = integral(delay:length(integral));
                % QRS Arama
                max_h = max(integral);
                thresh = 0.2;
                poss_reg = integral > (thresh*max_h);

                sol = find(diff([0 poss_reg'])==1);
                sag = find(diff([poss_reg' 0])==-1);
 
                 for m=1:length(sol)
                    [maxdeger(m) maxloc(m)] = max( comb(sol(m):sag(m)) );
                    maxloc(m) = maxloc(m)-1+sol(m); % offset ekle
                    [mindeger(m) minloc(m)] = min( comb(sol(m):sag(m)) );
                    %mindeger Q noktasini verir
                    minloc(m) = minloc(m)-1+sol(m); % offset ekle
                    % Q noktasi icin offset
                 end

                maxpozisyon=ones(1,4000)*(-max(comb));
                maxpozisyon(1,maxloc)=max(comb);

%                 figure
%                 plot(comb)
%                 title('R noktalari belirlenmis sinyal');
%                 hold on
%                 plot(maxloc,maxdeger,'g-')
% 
%                 figure
%                 plot(comb)
%                 title('R noktalalari belirlenmiþ ECG sinyali');
%                 hold on
%                 plot(maxpozisyon,'r')
%                 legend('Comb Filtreden Gecen Sinyal','R');
                temp=1;
                x=min(maxloc);
                y=max(maxloc);
                %y=400-(j-max(maxloc));
                
                temp=ones(1,651);
                    for v=1:length(maxloc)
                        if(maxloc(1,v)-250>0 && maxloc(1,v)+400<length(comb))
                            s=comb(maxloc(1,v)-250:maxloc(1,v)+400);
                            %temp=temp.*s;
                        end
                    end
                  %  s=temp;
               [M,I] = max(s(:));
              % I=I/2;
                % Compute coefficients COEFS using cwt
                COEFS = cwt(s,1:89,'coif1');
                coefs_rsize=imresize(COEFS,[128 128]);

                % Compute and plot the scalogram (image option)
                figure;
                SC = wscalogram('image',coefs_rsize);
                
                %Convolution3x3 ve Pooling2x2
                SC_Conv=convolution_sb(SC);
                Pooling=pooling_sb(SC_Conv);
                
                SC_Conv=convolution_sb(Pooling);
                Pooling=pooling_sb(SC_Conv);
                
                SC_Conv=convolution_sb(Pooling);
                Pooling=pooling_sb(SC_Conv);
                
                SC_Conv=convolution_sb(Pooling);
                Pooling=pooling_sb(SC_Conv);
                
                temp=[];
                for t=1:length(Pooling)
                    temp=horzcat(temp,Pooling(t,:));
                end
                healthy(i,:)=temp;
                               
                
        end
%         figure
%         plot(ecg_smooth)
%         figure
%         plot(filtered_signal)
figure
subplot(2,1,1);
plot(ecg_lead)
xlim([0 5000]);

title('EKG Ýþareti');
xlabel('Zaman');ylabel('Genlik');

subplot(2,1,2);
plot(ecg_smooth)
xlim([0 5000]);

title('DC Bileþeni Filtrelenmiþ EKG Ýþareti');
xlabel('Zaman');ylabel('Genlik');

%%%%%
figure
subplot(2,1,1);
plot(ecg_smooth)
xlim([0 5000]);

title('DC Bileþeni Filtrelenmiþ EKG Ýþareti');
xlabel('Zaman');ylabel('Genlik');

subplot(2,1,2);
plot(filtered_signal)
xlim([0 5000]);

title('Filtrelenmiþ EKG Ýþareti');
xlabel('Zaman');ylabel('Genlik');
%%%%%
figure
subplot(2,1,1);
plot(ecg_smooth)
xlim([0 5000]);

title('EKG Ýþaretinin R noktalarý Genlik Seviyeleri');
xlabel('Zaman');ylabel('Genlik');

subplot(2,1,2);
plot(filtered_signal)
xlim([0 5000]);

title('R Noktalarý Belirlenmiþ EKG Ýþareti');
xlabel('Zaman');ylabel('Genlik');