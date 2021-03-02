
%let input_path = C:\Users\mehdi\Desktop\M2 MOSEF\Scoring\DRIM\sas_input ;
%let output_path = C:\Users\mehdi\Desktop\M2 MOSEF\Scoring\DRIM\sas_output ;

/************************************* Importation des bases *************************************/

%macro import(mat);
proc import datafile = "&input_path\mat&mat.nodiscr.csv"
 out = mat&mat.
 dbms = csv
 replace;
run;

%mend import;

%import(6) ; %import(9) ; %import(12) ; %import(18) ; %import(24) ; 

/************************************* Enlever l'index et la colonne tx_rec_marg continue *************************************/

%macro clean(mat);
data dfm&mat ;
set mat&mat;
drop tx_rec_marg var1;
run;
%mend clean ; 

%clean(6) ; %clean(9) ; %clean(12) ; %clean(18) ; %clean(24) ; 


/************************************ TRAIN TEST SPLIT *************************************/

%macro train_test_split(mat);
proc surveyselect data=dfm&mat. method=srs seed=43543 outall
  samprate=0.7 out=subset&mat.;

data training&mat.;
   set subset&mat.;
   if Selected=1;
  
  
data test&mat.;
   set subset&mat.;
   if Selected=0;
%mend ;

%train_test_split(6) ; %train_test_split(9) ; %train_test_split(12) ; %train_test_split(18) ; %train_test_split(24) ;



/* ******************************************************** PROC LOGISTIC CONFUSION MATRICES ON TEST SETS *****************************************************/

PROC LOGISTIC DATA=Training6  outmodel=mat6Model DESCENDING   ;
	class   tx_rec_marg_Bin (ref = "0")  CD_CAT_EXPO_4 (ref = "0") date_neg (ref = "0") fusion (ref = "0")  / param=ref ;
	MODEL tx_rec_marg_Bin  = CD_CAT_EXPO_4 ratio_ead date_neg fusion DUR_PREV_FIN MT_INI_FIN_ mt_appo__CD_CAT_EXPO_4
									DUR_PREV_FIN_CD_CAT_EXPO_4 ead_cat_seg ead_qual_veh/LINK=gLOGIT   ; 
	weight weight;
	output out=preds predprobs=individual;
	score data=Test6 out=probas6;
RUN;
QUIT;
data probas6 ;
set probas6;
label I_tx_rec_marg_Bin = "Valeur prédite du taux de recouvrement" ;
run;
proc freq data=probas6;
table tx_rec_marg_Bin *I_tx_rec_marg_Bin / out=CellCounts;
run;
data CellCounts;
set CellCounts;
Match=0;
if tx_rec_marg_Bin=I_tx_rec_marg_Bin then Match=1;
run;
proc means data=CellCounts mean;
freq count;
var Match;
run;

proc export data=probas6 
  dbms=xlsx 
  outfile="&output_path.\probas6" 
  replace;
run;


PROC LOGISTIC DATA=Training9 outmodel=mat9Model DESCENDING;
	class    tx_rec_marg_Bin (ref = "0") cat_seg (ref = "0") qual_veh (ref = "0") fusion (ref = "0") date_neg (ref = "0")/ param=ref ;
	MODEL tx_rec_marg_Bin  =  CD_CAT_EXPO_4 cat_seg qual_veh ratio_ead date_neg fusion DUR_PREV_FIN MT_INI_FIN_ chom_spain
								cli_spain mt_appo__cat_seg mt_appo__CD_CAT_EXPO_4 ead_cat_seg ead_qual_veh /LINK=gLOGIT ;
	weight weight;
	output out=preds predprobs=individual;
	score data=Test9 out=probas9; 

RUN;
QUIT;
data probas9 ;
set probas9;
label I_tx_rec_marg_Bin = "Valeur prédite du taux de recouvrement" ;
run;
proc freq data=probas9;
table tx_rec_marg_Bin *I_tx_rec_marg_Bin / out=CellCounts;
run;
data CellCounts;
set CellCounts;
Match=0;
if tx_rec_marg_Bin=I_tx_rec_marg_Bin then Match=1;
run;
proc means data=CellCounts mean;
freq count;
var Match;
run;

proc export data=probas9 
  dbms=xlsx 
  outfile="&output_path.\probas9" 
  replace;
run;



PROC LOGISTIC DATA=Training12  outmodel=mat12Model  DESCENDING;
	class  tx_rec_marg_Bin (ref = "0") fusion (ref = "0")  date_neg (ref = "0") / param=ref ;
	MODEL tx_rec_marg_Bin  = ratio_b_endm_ ratio_ead date_neg fusion DUR_PREV_FIN mt_appo__cat_seg DUR_PREV_FIN_CD_CAT_EXPO_4 ead_cat_seg  /LINK=gLOGIT  ;
	weight   weight ;
	output out=preds predprobs=individual;
	score data=Test12 out=probas12; 
RUN;
QUIT;
data probas12 ;
set probas12;
label I_tx_rec_marg_Bin = "Valeur prédite du taux de recouvrement" ;
run;
proc freq data=probas12;
table tx_rec_marg_Bin *I_tx_rec_marg_Bin / out=CellCounts;
run;
data CellCounts;
set CellCounts;
Match=0;
if tx_rec_marg_Bin=I_tx_rec_marg_Bin then Match=1;
run;
proc means data=CellCounts mean;
freq count;
var Match;
run;

proc export data=probas12 
  dbms=xlsx 
  outfile="&output_path.\probas12" 
  replace;
run;



PROC LOGISTIC DATA=Training18  outmodel=mat18Model DESCENDING;
    class    tx_rec_marg_Bin (ref = "0") date_neg (ref = "0")  fusion (ref = "0") no_appo (ref = "0")/ param=ref ;
	MODEL tx_rec_marg_Bin  = CD_CAT_EXPO_4 pct_appo_ qual_veh ratio_b_endm_ no_appo date_neg fusion DUR_PREV_FIN MT_INI_FIN_ mt_appo__CD_CAT_EXPO_4 ead_qual_veh  /LINK=gLOGIT   ;  
	weight weight ;
	output out=preds predprobs=individual;
	score data=test18 out=probas18; 

RUN;
QUIT;
data probas18 ;
set probas18;
label I_tx_rec_marg_Bin = "Valeur prédite du taux de recouvrement" ;
run;
proc freq data=probas18;
table tx_rec_marg_Bin *I_tx_rec_marg_Bin / out=CellCounts;
run;
data CellCounts;
set CellCounts;
Match=0;
if tx_rec_marg_Bin=I_tx_rec_marg_Bin then Match=1;
run;
proc means data=CellCounts mean;
freq count;
var Match;
run;


proc export data=probas18 
  dbms=xlsx 
  outfile="&output_path.\probas18" 
  replace;
run;




PROC LOGISTIC DATA=Training24 outmodel=mat24Model DESCENDING;
    class   tx_rec_marg_Bin (ref = "0")  CD_CAT_EXPO_4 (ref = "0") fusion (ref = "0") date_neg (ref = "0") / param=ref ;
	MODEL tx_rec_marg_Bin  = CD_CAT_EXPO_4 pct_appo_ date_neg fusion DUR_PREV_FIN MT_INI_FIN_ chom_spain MT_INI_FIN__CD_CAT_EXPO_4 mt_appo__cat_seg
							mt_appo__CD_CAT_EXPO_4 ead_cat_seg ead_qual_veh/LINK=gLOGIT  ;  
	weight weight;
	output out=preds predprobs=individual;
	score data=Test24 out=probas24; 
RUN;
QUIT;
data probas24 ;
set probas24;
label I_tx_rec_marg_Bin = "Valeur prédite du taux de recouvrement" ;
run;
proc freq data=probas24;
table tx_rec_marg_Bin *I_tx_rec_marg_Bin / out=CellCounts;
run;
data CellCounts;
set CellCounts;
Match=0;
if tx_rec_marg_Bin=I_tx_rec_marg_Bin then Match=1;
run;
proc means data=CellCounts mean;
freq count;
var Match;
run;

/****************************************** EXPORT Logistic results ************************************************/

%macro export(mat)
proc export data=probas24 
  dbms=xlsx 
  outfile="&output_path.\probas&mat." 
  replace;
run;
%mend;

%export(6); %export(9); %export(12); %export(18); %export(24); 
