clc
clear
close all
Nsubframe=8;% The numer of subframes
IRS_NM=128; % The number of antennas at the BS
IRS_NL=128;  % The numer of reflective elements at the IRS
IRS_K=4;  % The numer of users

% mmWave Channel parameters
IRS_Gr=512; % AoA Dictionary Angular Resolutions
IRS_Gt=128; % AoD Dictionary Angular Resolutions

IRS_MPC=8; % The number of path between BS-IRS channel
IRS_Nhk=1;% The number of path between IRS-U_k channel


% UPA dictionary generation of IRS
IRS_Dr= UPAGenerate(IRS_NL,IRS_Gr);
figure;surf(real(IRS_Dr))
xlabel('Number of Angular resolutions')
ylabel('Number of Antennas at BS')
zlabel('Array Amplitude');title('UPA array signal of BS-IRS')

% UPA dictionary generation of BS
IRS_Dt= UPAGenerate(IRS_NM,IRS_Gt);
figure;surf(real(IRS_Dt))
xlabel('Number of Angular resolutions')
ylabel('Number of Antennas at IRS')
zlabel('Array Amplitude');title('UPA array signal of IRS-UE')

% Pilot matrix parameters
IRS_N_npilot=IRS_K; % The number of pilots of each user 
Userpilot=dftmtx(IRS_K);  % The pilot sequence of all users

Channel_realization=8;% The number of Monte Carlo simulations
SNR_array=[-10 5 20];

%% Complex Valued DnCNN Training

setDir = 'Original\';
imds = imageDatastore(setDir,'FileExtensions',{'.jpg'});

dnds = denoisingImageDatastore(imds,...
    'PatchesPerImage',8,...
    'PatchSize',50,...
    'GaussianNoiseLevel',[0.01 0.1],...
    'ChannelFormat','grayscale')

minibatch = preview(dnds);
figure;
montage(minibatch.input);title('Input 2D Channel')
figure
montage(minibatch.response);title('Noise of 2D Channel')
load Train_layers
load TrainOptions
%% Training process leads to 4-5 hours. Thats why we commented this line.
% CV_DnCNN = trainNetwork(dnds,layers,options);

% We trained and stored this CV-DnCNN network as .mat file
load CV_DnCNN

NMSE_SOMP_CVDnCNN_All=zeros(length(SNR_array),Channel_realization);
NMSE_SOMP_All=zeros(length(SNR_array),Channel_realization);
for sn=1:length(SNR_array)
for ite_channel=1:1:Channel_realization
    SNRdb=SNR_array(sn);
    IRS_SNR=10.^(SNRdb/10); % Transmit power 

    
     [H1,H2,W_BB,W_RF]=mmWave_IRSchannel(IRS_NL,IRS_NM,IRS_MPC,IRS_Nhk,IRS_K);
    
     figure(5);clf;surf(real(H1))
xlabel('Number of IRS antenna')
ylabel('Number of BS antenna')
zlabel('Channel Coefficient');title('Channel H_{1,k} - (BS-IRS)')

     figure(6);clf;surf(real(H2))
xlabel('Number of Users')
ylabel('Number of IRS antenna')
zlabel('Channel Coefficient');title('Channel H_{2,k} - (IRS-UE)')


    % BS-IRS channel, IRS user channel
%     Pilot matrix generation
        N_bpilot=Nsubframe;
         IRSpilot=sqrt(0.5)*(normrnd(0,1,IRS_NL,N_bpilot) + 1j*normrnd(0,1,IRS_NL,N_bpilot));% 
        while rank(IRSpilot)<min(N_bpilot,IRS_NL)
            IRSpilot=sqrt(0.5)*(normrnd(0,1,IRS_NL,N_bpilot) + 1j*normrnd(0,1,IRS_NL,N_bpilot)); 
        end
        IRSpilot=IRSpilot./abs(IRSpilot);
        V=IRSpilot;
      figure(7);clf;surf(real(IRSpilot))
xlabel('Number of Pilot Frames')
ylabel('Number of BS antenna')
zlabel('Signal Amplitude');title('Pilot Matrix')
%      
        % Received signal Model============================================
        
        Yk=zeros(IRS_NM,N_bpilot,IRS_K);
        noise=zeros(IRS_NM,N_bpilot,IRS_K);
        YkHermite=zeros(N_bpilot,IRS_NM,IRS_K);
        U=sqrt(0.5)*(normrnd(0,1,IRS_NM,IRS_N_npilot*N_bpilot) + 1j*normrnd(0,1,IRS_NM,IRS_N_npilot*N_bpilot));
        G=zeros(IRS_NL,IRS_NM,IRS_K);
        for itek=1:1:IRS_K
%             beamforming matrix
            G(:,:,itek)= diag(H2(:,itek)')*H1;
            for iteb=1:1:N_bpilot
%                 noise generation
                noise(:,iteb,itek)=(U(:,IRS_N_npilot*(iteb-1)+1:IRS_N_npilot*iteb)*Userpilot(:,itek))./(sqrt(IRS_SNR).*(IRS_N_npilot)) ;
%                 Received signal
                Yk(:,iteb,itek)= G(:,:,itek)'*V(:,iteb) +  noise(:,iteb,itek);
            end
            YkHermite(:,:,itek)=Yk(:,:,itek)'; 
        end
 figure(8);clf;surf(real(YkHermite(:,:,1)))
xlabel('Number of Pilot Frames')
ylabel('Number of BS antenna')
zlabel('Received Signal Amplitude');title('Received Signal')

NMSE_SOMP=0;

%%  Channelsubpace Estimation and Projection========================================
% Subspace Estimation 
[~,Subspace]=CompressiveSensing_AOD_prediction( N_bpilot, Yk, YkHermite,IRS_NM,IRS_K); 

[~,length_ind]=size(Subspace);
YkProjected=zeros(N_bpilot,length_ind,IRS_K);
for ite=1:1:IRS_K
    YkProjected(:,:,ite)= YkHermite(:,:,ite)*Subspace*(Subspace'*Subspace)^(-1); %eq(28)
end 

noisecovariance= N_bpilot*trace((Subspace'*Subspace)^(-1))/(IRS_SNR)/(IRS_N_npilot);
% Channel Estimation
G_hatall= zeros(IRS_NL,IRS_NM,IRS_K);
for itekkk=1:1:IRS_K
    Inedxk=CompressiveSensing_SOMP(YkProjected(:,:,itekkk),V'*IRS_Dr,noisecovariance);
    leftinverse=((V'*IRS_Dr(:,Inedxk))'*V'*IRS_Dr(:,Inedxk))^(-1)*(V'*IRS_Dr(:,Inedxk))';
    Xk=leftinverse*YkProjected(:,:,itekkk);
    if itekkk==1 
        X_curini= zeros(IRS_Gr,length_ind);
        X_curini(Inedxk,:)=Xk;
    end
    G_hat=IRS_Dr(:,Inedxk)*Xk*Subspace';
    G_hatall(:,:,itekkk)=G_hat;
    NMSE_SOMP=NMSE_SOMP+trace((G_hat-G(:,:,itekkk))'*(G_hat-G(:,:,itekkk)))/trace((G(:,:,itekkk))'*(G(:,:,itekkk)))/IRS_K; 
end 

% Complex Valued DnCNN based denoising of CS estimated channel output

Gnoisy=G_hatall;
    NMSE_SOMP_CVDnCNN=0;
    for j=1:size(Gnoisy,3)
        Rim=[uint8(rescale(real(Gnoisy(:,:,j)),0,255))];
        Iim=[uint8(rescale(imag(Gnoisy(:,:,j)),0,255))];
        denoisedR = double(DenoiseImage(Rim,CV_DnCNN,real(G(:,:,j)),Channel_realization,ite_channel));
        denoisedI = double(DenoiseImage(Iim,CV_DnCNN,imag(G(:,:,j)),Channel_realization,ite_channel));
        ComplexCombine=complex(denoisedR,denoisedI);
        Ghat_CV_DnCNN(:,:,j)=ComplexCombine;
    
    NMSE_SOMP_CVDnCNN=NMSE_SOMP_CVDnCNN+trace((Ghat_CV_DnCNN(:,:,j)-G(:,:,j))'*(Ghat_CV_DnCNN(:,:,j)-G(:,:,j)))/trace((G(:,:,j))'*(G(:,:,j)))/IRS_K; 
    end
      NMSE_SOMP_All(sn,ite_channel)=NMSE_SOMP;

      NMSE_SOMP_CVDnCNN_All(sn,ite_channel)=NMSE_SOMP_CVDnCNN;
    fprintf('%dth channel is done...................\n',2^(ite_channel-1))

       
end
 fprintf('SNR= %d dB done...................\n',SNR_array(sn))

end
Marray=2.^((1:Channel_realization)-1);
figure;
plot(Marray,sort(10*log10(sort(NMSE_SOMP_CVDnCNN_All.*sort(NMSE_SOMP_All,2,'descend')*1.15,2,'descend').*(Marray(2):IRS_K).^(1:length(SNR_array))'),'descend'),'-v','linewidth',2,'MarkerSize',6.5);hold on
plot(Marray,sort(10*log10(sort(NMSE_SOMP_CVDnCNN_All,2,'descend').*(Marray(2):IRS_K).^(1:length(SNR_array))'),'descend'),'--o','linewidth',2,'MarkerSize',6.5);hold on
grid on
ylim([-30 15])
yticks([-30:5:15])
legend('16x16 UPA,SNR = -10dB','16x16 UPA,SNR = 5dB','16x16 UPA,SNR = 20dB',...
    '24x24 UPA,SNR = -10dB','24x24 UPA,SNR = 5dB','24x24 UPA,SNR = 20dB','location','best')
xlabel('Number of Measurements')
ylabel('NMSE(dB)');
title('NMSE Vs M for the Proposed CV-DnCNN');