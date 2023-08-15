function  [H1,H2,W_BB,W_RF]=mmWave_IRSchannel(NL,NM,Nf,Nhk,K)
 
 
% Precoder of mmwave transmitter (BS-IRS)
F_precoder=zeros(NL,Nf) ;
% Combiner of mmwave receiver
W_RF=zeros(NM,Nf);
% Precoder of mmwave transmitter (IRS-UE)
F_precoderHk=zeros(NL,Nhk,K);
% Combiner of mmwave receiver
W_BB=zeros(NL,Nhk*Nf,K); 
% BS-RIS Channel
H1=zeros(NL,NM);
% Analog precoder
theta_nmAOA= ((rand(1,Nf)*pi)-pi/2); 
% Digital precoder
theta_nmAOD=((rand(1,Nf)*pi)-pi/2); 

% Hybrid beamforming
for iteaoa=1:1:length(theta_nmAOA)
    F_precoder(:,iteaoa)=(exp(-1j*pi*sin(theta_nmAOA(iteaoa))*[0:NL-1])).';
    W_RF(:,iteaoa)=(exp(-1j*pi*sin(theta_nmAOD(iteaoa))*[0:NM-1])).';
    H1=H1+sqrt(0.5)*(normrnd(0,1,1,1) + 1j*normrnd(0,1,1,1)).*(F_precoder(:,iteaoa)* W_RF(:,iteaoa)');
end
H1=sqrt(1/Nf).*H1;

% RIS-User Channel
H2=zeros(NL,K);
for itek=1:1:K
    theta_User=(rand(K,Nhk)*pi)-pi/2;%[10 30 ]';
    H2(:,itek)=zeros(NL,1);
    for iteuaoa=1:1:Nhk
        F_precoderHk(:,iteuaoa,itek)=((exp(-1j*pi*sin(theta_User(itek,iteuaoa))*[0:NL-1])).');
        W_BB(:,(iteuaoa-1)*length(theta_nmAOA)+1:(iteuaoa)*length(theta_nmAOA),itek)=diag(F_precoderHk(:,iteuaoa,itek)')*F_precoder;
        H2(:,itek)= H2(:,itek)+sqrt(0.5)*(normrnd(0,1,1,1) + 1j*normrnd(0,1,1,1)).*F_precoderHk(:,iteuaoa,itek);
    end
    H2(:,itek)= sqrt(1/(Nhk)).*H2(:,itek);
end
end