% ------------------clasificare_SVM.m----------------------- %
%{
clear
data = load('wksp_XY.mat');
puncteTotale = [df.X(:,1) df.X(:,2)]
etichete = grp2idx(data.Y);
etichete(etichete==2) = -1;

figure;
plot(puncteClasif1(:,1),puncteClasif1(:,2),'k.','MarkerSize',15)
hold on
plot(puncteClasif2(:,1),puncteClasif2(:,2),'r.','MarkerSize',15)
ezpolar(@(x)1);ezpolar(@(x)sqrt(5));
axis equal
hold off
%}

% pregatire instante de intrare intr-o singura matrice precum si pregatirea vectorului iesirilor aferente celor doua clase
data = load('wksp_XY.mat');
puncteTotale = [data.X(:,1) data.X(:,2)];
etichete = grp2idx(data.Y);
etichete(etichete==2) = -1;

% antreneaza clasificator SVM
clasifSVM = fitcsvm(puncteTotale,etichete,'KernelFunction','rbf','BoxConstraint',Inf,'ClassNames',[-1,1]);

% construieste gridul de predictie pentru aflarea hipersufprafetei de decizie
pas = 0.02;
[xx,yy] = meshgrid(min(puncteTotale(:,1)):pas:max(puncteTotale(:,1)), min(puncteTotale(:,2)):pas:max(puncteTotale(:,2)));
reteaPuncte = [xx(:),yy(:)]; % pregatire puncte grid pentru predictie la apartenenta de clasa
[~,clasePredictate] = predict(clasifSVM,reteaPuncte);

% afiseaza datele precum si hipersuprafata de decizie
figure;
h(1:2) = gscatter(puncteTotale(:,1),puncteTotale(:,2),etichete,'kr','.');
hold on
ezpolar(@(x)1);
h(3) = plot(puncteTotale(clasifSVM.IsSupportVector,1),puncteTotale(clasifSVM.IsSupportVector,2),'ko');
contour(xx,yy,reshape(clasePredictate(:,2),size(xx)),[0 0],'k'); % afiseaza curba de nivel 0
legend(h,{'-1','+1','Vectori suport'});

% calcul scor de apartenenta cu formula din help, pentru ultima instanta
% din reteaua de puncte reteaPuncte(end,:)
diferente_=[puncteTotale(clasifSVM.IsSupportVector,1)-reteaPuncte(end,1)];
diferente=[diferente_ puncteTotale(clasifSVM.IsSupportVector,2)-reteaPuncte(end,2)];
kernele=exp(-(diferente(:,1).^2+diferente(:,2).^2));
sum(etichete(clasifSVM.IsSupportVector).*clasifSVM.Alpha.*kernele) + clasifSVM.Bias;
clasePredictate(end,:);

% ---------regresie logistica neliniara-------
% re-etichetare apartenenta clase ajustate logit(y)

%{
etichete = 0.99*ones(200,1); %
etichete(1:100) = 0.01;
% utilizeaza modelul de regresie logit(y)=1+theta1*x1+...
mdlRegLogNel=fitglm(puncteTotale,etichete,'quadratic','link','logit')

pas = 0.02;
[xx,yy] = meshgrid(min(puncteTotale(:,1)):pas:max(puncteTotale(:,1)),...
min(puncteTotale(:,2)):pas:max(puncteTotale(:,2)));
reteaPuncte = [xx(:),yy(:)]; % pregatire puncte grid pentru predictie la apartenenta de clasa
[clasePredictate,~] = predict(mdlRegLogNel,reteaPuncte);
%}

% afiseaza instantele precum si hipersuprafata de decizie
figure;
gscatter(puncteTotale(:,1),puncteTotale(:,2),etichete,'kr','.');
hold on,ezpolar(@(x)1);hold on,
contour(xx,yy,reshape(clasePredictate(:,1),size(xx)),[0.5 0.5],'k');
legend(h,{'-1','+1','Vectori suport'});
%--------------------------------------------------------------------------------------%
