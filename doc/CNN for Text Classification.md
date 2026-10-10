ProceedingsoftheTwenty-SeventhInternationalJointConferenceonArtificialIntelligence(IJCAI-18)
Transformable Convolutional Neural Network for Text Classification
LiqiangXiao1;2,HonglunZhang1;2,WenqingChen1;2,YongkunWang3,YaohuiJin1;2 (cid:3)
1 StateKeyLabofAdvancedOpticalCommunicationSystemandNetwork,
ShanghaiJiaoTongUniversity
2
ArtificialIntelligenceInstitute,ShanghaiJiaoTongUniversity
3 NetworkandInformationCenter,ShanghaiJiaoTongUniversity
{xiaoliqiang,zhanghonglun,wenqingchen,ykw,jinyh}@sjtu.edu.cn
|     |     | Abstract |     |     |     | oversizefeatures.Inanotherword,itcannotfittheshape |     |     |     |     |     |     |     |
| --- | --- | -------- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
offeatures.
Convolutionalneuralnetworks(CNNs)haveshown
|                 |             |     |             |          |     | • Traditional | CNNs |     | are unable | to  | actively | adjust | them- |
| --------------- | ----------- | --- | ----------- | -------- | --- | ------------- | ---- | --- | ---------- | --- | -------- | ------ | ----- |
| their promising | performance |     | for natural | language |     |               |      |     |            |     |          |        |       |
processingtasks,whichextractn-gramsasfeatures selves to adapt to the transformation of features. They
to represent the input. However, n-gram based tendtorememberallkindsofvariantsoffeaturesfroma
largedataset,butnottolearnthepatternsoftransforma-
| CNNs are | inherently | limited | to fixed | geometric |     |       |                                           |     |     |     |     |     |     |
| -------- | ---------- | ------- | -------- | --------- | --- | ----- | ----------------------------------------- | --- | --- | --- | --- | --- | --- |
|          |            |         |          |           |     | tion. | Thisisundesirablebecauseonefeaturemayhave |     |     |     |     |     |     |
structureandcannotproactivelyadapttothetrans-
alotoftransformedforms.
| formations      | of features. | In  | this paper,          | we  | propose   |              |               |         |        |         |             |           |         |
| --------------- | ------------ | --- | -------------------- | --- | --------- | ------------ | ------------- | ------- | ------ | ------- | ----------- | --------- | ------- |
| two modules     | to provide   |     | CNNs with            | the | flexibil- |              |               |         |        |         |             |           |         |
|                 |              |     |                      |     |           | For example, | given         | a       | phrase | of “not | nearly      | as good”, | tra-    |
| ity for complex | features     |     | and the adaptability |     | for       |              |               |         |        |         |             |           |         |
|                 |              |     |                      |     |           | ditional     | convolutional | network |        | is hard | to directly |           | capture |
transformation,namely,transformableconvolution the non-consecutive pattern “not...good”. It is also difficult
and transformable pooling. Our method fuses dy- to recognize this pattern from numerous transformed forms,
namicandstaticdeviationstoredistributethesam- suchas“notsogood”,“notevengood”.
plinglocations,whichcancapturebothcurrentand Traditionally, most works cope with above challenges by
| globaltransformations. |     | Ourmodulescanbeeasily |     |     |     |            |            |     |          |          |      |          |     |
| ---------------------- | --- | --------------------- | --- | --- | --- | ---------- | ---------- | --- | -------- | -------- | ---- | -------- | --- |
|                        |     |                       |     |     |     | increasing | the volume | of  | datasets | to cover | more | variants | of  |
integrated by other models to generate new trans- features or employ the constant features [Lowe, 1999; Bay
formable networks. We test proposed modules on et al., 2006; Rublee et al., 2011]. However, the potential of
twostate-of-the-artmodels,andtheresultsdemon- thesetwoapproachesislimitedbyrequiringtoomuchlabor
stratethatourmodulescaneffectivelyadapttothe workordecreasingtheperformance.
featuretransformationintextclassification.
|     |     |     |     |     |     | Similar | challenges | also | exist | in the | field of | computer | vi- |
| --- | --- | --- | --- | --- | --- | ------- | ---------- | ---- | ----- | ------ | -------- | -------- | --- |
sion. Someworksfocusoninherentlyaugmentingtheflexi-
|     |     |     |     |     |     | bility of | CNN models | and | the | adaptability | for | feature | trans- |
| --- | --- | --- | --- | --- | --- | --------- | ---------- | --- | --- | ------------ | --- | ------- | ------ |
1 Introduction
|     |     |     |     |     |     | formation, | which | allow | the models | to  | reshape | some | lay- |
| --- | --- | --- | --- | --- | --- | ---------- | ----- | ----- | ---------- | --- | ------- | ---- | ---- |
CNNs have been proven effective in many nature language ers to offset the feature transformation [Jia et al., 2012;
processing tasks such as representation learning [Liu et al., Jaderbergetal.,2015;JeonandKim,2017;Daietal.,2017].
2015],informationretrieval[Shenetal.,2014],textclassifi-
Theycanbedividedintotwothreads:1)learningstaticshape
cation[Kalchbrenneretal.,2014],etc. Convolutionallayers parameters through back-propagation to capture only global
intheCNNsextractfeatureswithgeometricallyfixedfilters, distributionoftransformationpatterns[JeonandKim,2017;
whichcanberegardedasanimplementationofN-Gramlan- Jaderbergetal.,2015];2)learningdynamicalshapeinforma-
guagemodel.
tionaccordingtotheinputstoobtainonlycurrenttransforma-
However, regular CNNs inevitably face the challenge of tionpatterns[Daietal.,2017].However,thesemodelscannot
adapting to the feature transformation, which is caused by bedirectlyappliedtotheCNNsdesignedforNLPtasks.
thefactthatCNNshavemanygeometricallyfixedstructures, Inthispaper,wecombinethethoughtsofboththreadsand
such as the convolutional layer limiting the filter shape into proposestwoend-to-endmodulesfortextclassificationtore-
n-gram, the pooling layer selecting the superior features by visetheconvolutionandpoolinglayersrespectively. Bothof
| solidchunks. | Thesefixedstructuresrendertwomainlimita- |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------ | ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ourmodulescanenhancethecapabilityofCNNsformodel-
tionsforlanguagerepresentation: ingthetransformationpatterns,namely,TransformableCon-
| •               |     |        |                |     |            | volution | and Transformable |     | Pooling. | Transformable |     |     | convo- |
| --------------- | --- | ------ | -------------- | --- | ---------- | -------- | ----------------- | --- | -------- | ------------- | --- | --- | ------ |
| All the filters | and | chunks | are continuous |     | and shape- |          |                   |     |          |               |     |     |        |
lutionadds1-Dimension(1D)deviationstothesamplinglo-
| fixed, which | make | it difficult | for CNNs |     | to deal with |            |          |       |         |           |         |     |      |
| ------------ | ---- | ------------ | -------- | --- | ------------ | ---------- | -------- | ----- | ------- | --------- | ------- | --- | ---- |
|              |      |              |          |     |              | cations of | filters, | which | can not | only help | capture | the | com- |
somecomplexscenariossuchasthenon-consecutiveor
|     |     |     |     |     |     | plex features | but | also offset | the | transformation |     | of  | features. |
| --- | --- | --- | --- | --- | --- | ------------- | --- | ----------- | --- | -------------- | --- | --- | --------- |
(cid:3)Correspondingauthor
|     |     |     |     |     |     | The deviations | are | automatically |     | learned | without | needs | for |
| --- | --- | --- | --- | --- | --- | -------------- | --- | ------------- | --- | ------- | ------- | ----- | --- |
4496

dynamic deviation
ProceedingsoftheTwenty-SeventhInternationalJointConferenceonArtificialIntelligence(IJCAI-18)
Output feature map is generated by a convolving operation
conv Static Dynamic deviations betweenweightwandinputx,whichcanbeformulatedas
deviations
d×l×k/w
y=w∗x: (2)
Δp
i
Dc
Andeveryelementinfeaturemapycanbecalculatedby:
∑
y(p )= w(p )·x(p +p ); (3)
p 0 i 0 i
0 pi 2C
Δp i Sc wherep 0 isalocationinoutputfeaturemapandp i enumer-
input sequence/ feature map atesthepointsincollectionCofsamplinglocations.
output feature map
l×k back-propagation l×k/w Intransformableconvolution,Cisdividedintotwoparts,
Static deviations one is current feature related, and the other is globally de-
s×1×1 pended. Let D ⊂ C and S ⊂ C denote them respec-
c c
Figure 1: Illustration of Transformable Convolution. l;k;d;s and tively. Then they are added with different deviations ∆pD i c
wrepresentthelengthoffeaturemap,widthoffeaturemap,sizeof and ∆pSc to revise the sampling locations. And Eq.(3) be-
i
D c ,sizeofS c andthewidthoffiltersrespectively. Thewidthkof comes ∑
theoutputfeaturemapisdividedbywbecauseconvolutionfortext
is1D. y(p 0 )= w(p i )·x(p 0 +p i +∆pD i c)
pi 2D∑c
extrasupervisionorhumanexperience. Transformablepool- + w(p )·x(p +p +∆pSc): (4)
i 0 i i
ingsharesthesameideawithtransformableconvolution. On
pi 2Sc
the base of pooling layer, it adds 1D deviations to the sam-
Now,samplinglocationsoffilterareredistributed,andare
plinglocationsofchunkstoincreasetheadaptabilityforfea-
notintheshapeofregularrectangularanylonger. Consider-
tureselection.
Our modules can be easily embedded into other models ingthat∆pD i c and∆pS i c areoftenfractional, valueoffunc-
toconstructnewtransformableConvNets(TF-ConvNets)by
tionx(·)iscalculatedbylinearinterpolation:
∑
displacingtheircounterparts. Extensiveexperimentsandvi-
x(p)= K(p;q)·x(q): (5)
sualizationsontextclassificationdatasetsverifytheeffective-
nessofTF-ConvNetsforcapturingthepatternsoftransforma- q
tion. Herepdenotesp +p +∆pDcorp +p +∆pSc,qenumerates
0 i i 0 i i
the locations in the whole input feature map, and K is the
2 TransformableConvolutionandPooling
linearinterpolationkernel. Inlinearspace,K iscomputedby
In this section, we introduce the mechanisms of two trans-
formable modules, which have learnable shapes to adapt to K(p;q)=max(0;1−|p−q|); (6)
the changes of features [Dai et al., 2017]. Conventionally, wheremax(·)functionsasa“mask”,limitingthedilationof
the shapes of filters and chunks in convolution and pooling theinterpolationareainto1.
are fixed from the start. But the transformable modules add Figure 1 shows the mechanism of transformable convolu-
thelearnedpositiondeviationsontofiltersorchunks,making tion. The dynamic deviations ∆pDc at the top are learned
their shapes flexible and adaptable. Position deviations are i
from current input sequence or feature map via a plain con-
composedofdynamicpartandstaticpart. Atinferencestage,
volutionlayer,whosevaluesareaddedtosamplinglocations
the dynamic deviations are related to current input, whose
inD . Sincetheyaregeneratedfromthepreviouslayer,their
c
values are actively learned from current features to capture
values change dynamically according to the current features
thecurrenttransformationinformation. Onthecontrary,val-
to fit current transformation patterns. The static deviations
uesofstaticdeviationsareupdatedbyback-propagationand ∆pSc at bottom are variables, which are updated by back-
keep static at interference stage, which describe the global i
propagationduringtrainingstageandkeepsstaticatinference
distributionofshapeinformation.
time. Hence, thestaticdeviationsdescribeaglobaldistribu-
2.1 TransformableConvolution tionoffeatureshape.
CNNsforNLPusuallyemploymorethanonefiltersorchan- 2.2 TransformablePooling
nels. Forsimplicity,weprovidetheformulationforthecase
Therearemanyversionsofpoolingoperation,whichisused
with one filter and channel, but it is easy to extend to more
to select the most important features and reduce the resolu-
complexcases.
tion.HereweimproveaplainversionofChunk-MaxPooling
Intypicalconvolution,theshapeoffiltercanbedescribed
(Figure2),whichhasbeenprovenusefultokeeptheposition
by the center position p and the sampling locations p , the
0 i informationoffeatures.
relativedistancesfromp . WeuseCtorepresentthecollec-
0 Givenafeaturemapxofsizel×k, Chunk-MaxPooling
tionofp . Forinstance,whenthelengthofafilterinconven-
i firstdividesxintom×n(0 < m < l;0 < n ≤ k)1 chunks.
tionalconvolutionis5,
C={−2;−1;0;1;2}: (1) 1Typically,nequalskintextclassificationtasks.
4497

max
max
dynamic deviation
output feature map
ProceedingsoftheTwenty-SeventhInternationalJointConferenceonArtificialIntelligence(IJCAI-18)
max
Static
|     |     |     |     |     |     |     |     |     | pooling |     | fc  |            | Dynamic deviations |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | ---------- | ------------------ | --- | --- |
|     |     |     |     |     |     |     |     |     |         |     |     | deviations | e×m×n              |     |     |
ΔpDo
i
p
0
Figure2:IllustrationofplainChunk-MaxPooling.Theinputfeature
mapiscutintochunksandtheoutputfeaturemapisconstructedby output feature map
|                            |     |     |     |     |     |     |     |     |     |     |     | Δp So |     |     | m×n |
| -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- |
| themaxvaluesofthesechunks. |     |     |     |     |     |     |     |     |     |     |     | i     |     |     |     |
input sequence/ feature map
inputl× kfeature map
Then,itoutputsam×nfeaturemapybyconcatenatingthe Static   d e v iations
|                                                     |     |     |     |     |                        |     |     |          |                                     |     | f × 1 × 1 |     | back-propagation |          |     |
| --------------------------------------------------- | --- | --- | --- | --- | ---------------------- | --- | --- | -------- | ----------------------------------- | --- | --------- | --- | ---------------- | -------- | --- |
| maximumvalueofeverychunk.Letp                       |     |     |     |     | 0 denotetop-leftcorner |     |     |          |                                     |     |           |     |                  |          |     |
| ofthe(i;j)-thchunk,thepoolingprocesscanbeformulated |     |     |     |     |                        |     |     |          |                                     |     |           |     |                  | jD jandf |     |
|                                                     |     |     |     |     |                        |     |     | Figure3: | Illustrationoftransformablepooling. |     |           |     | e =              | o        | =   |
| as                                                  |     |     |     |     |                        |     |     | jS j.    |                                     |     |           |     |                  |          |     |
o
|     |     | y(p )=max({x(p |     |     | +p )}); |     | (7) |     |     |     |     |     |     |     |     |
| --- | --- | -------------- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | 0              |     | 0   | i       |     |     |     |     |     |     |     |     |     |     |
pi 2O
| where       | p   | enumer⌊ates⌋ | the | sampling | loc⌈atio⌉ns |      | ⌊O⌋ in |                                                   |     |     |     |     |     |     |     |
| ----------- | --- | ------------ | --- | -------- | ----------- | ---- | ------ | ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|             | i   |              |     |          |             |      |        | Inthepaper,weusenormalizedgradienttoproperlylimit |     |     |     |     |     |     |     |
| chunk(i;j), |     | and i        | l < | p <      | (i + 1)     | l ;j | k <    |                                                   |     |     |     |     |     |     |     |
⌈ ⌉ m i m n the movement of sampling locations. As shown in Eq. (9),
| p <(j+1) |     | k . |     |     |     |     |     | normalizedgradientisformalizedas |     |     |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| j        |     | n   |     |     |     |     |     |                                  |     |     |     |     |     |     |     |
Similar to transformable convolution, sampling locations (cid:12) (cid:12)
|     |     |     |     |     |     |     |     |     |     |     |     | /(cid:12) | (cid:12) |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | -------- | --- | --- |
are divided into two parts of D and S . Then we redis- @L @L @L
|     |     |     |     | o   | o   |     |     |     |     |     |     | (cid:12) | (cid:12)  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | --------- | --- | --- |
|     |     |     |     |     |     |     |     |     |     |     | =   | (cid:12) | (cid:12); |     | (9) |
tribute these sampling locations by adding the deviations @∆p @∆p @∆p
| ∆pDo;∆pSo |     | ∈ ∆p | to them | respectively. |     | This makes | the |     |     |     |     |     |     |     |     |
| --------- | --- | ---- | ------- | ------------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i         | i   |      |         |               |     |            |     |     |     |     |     |     |     |     |     |
filterfittheshapeoffeaturesbetter. Eq.(7)becomes whereListhelossfunctionand∆pdenotesallkindsofdevi-
|     |     |     |     |     |      |     |     | ations. Thismeansweonlyusethedirectionofgradientand |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | +pD | +∆pD | o)⊕ |     |                                                     |     |     |     |     |     |     |     |
y(p )= max (x(p o limitedthestrideupto1inoneiteration. Empirically,learn-
|     | 0   |     |     | 0   | i   | i   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
pD o2Do;pS o2So ingrateissetto0:001,whichindicatesthatthestrideisupto
|     |     | i    | i        |     |     |     |     |                      |     |     |     |     |     |     |     |
| --- | --- | ---- | -------- | --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- |
|     |     | +pSo | +∆pSo)); |     |     |     |     | 0:001inoneiteration. |     |     |     |     |     |     |     |
|     |     | x(p  |          |     |     |     | (8) |                      |     |     |     |     |     |     |     |
|     |     | 0    | i        | i   |     |     |     |                      |     |     |     |     |     |     |     |
where⊕denotestheconcatenationoperation.
InthesamewayasEq.(5)and(6),thevalueofxiscalcu- 3 TransformableConvNets
∆pDo
lated by 1D bilinear interpolation (Eq.(6)), since the Ourmodulesareelaboratelydesignedtobecompatiblewith
and∆pSo
aregenerallyfractional. existing models. They can be seamlessly used by replacing
| As                          | illustrated | in Figure | 3,  | the top                  | dynamic | deviations | are |                     |     |       |      |          |                 |     |      |
| --------------------------- | ----------- | --------- | --- | ------------------------ | ------- | ---------- | --- | ------------------- | --- | ----- | ---- | -------- | --------------- | --- | ---- |
|                             |             |           |     |                          |         |            |     | their counterparts, |     | since | they | have the | same interfaces |     | with |
| learnedfromthecurrentinput. |             |           |     | Moreconcretely,Chunk-Max |         |            |     |                     |     |       |      |          |                 |     |      |
thevanillaCNNsandareeasytotrainwithback-propagation
| Pooling | first | generates | a feature  | map,       | on which | a fully | con-   |            |     |               |     |       |             |      |     |
| ------- | ----- | --------- | ---------- | ---------- | -------- | ------- | ------ | ---------- | --- | ------------- | --- | ----- | ----------- | ---- | --- |
|         |       |           |            |            |          |         |        | algorithm. | In  | this section, | we  | embed | our modules | into | two |
| nected  | layer | emits the | normalized | deviations |          | ∆pˆ.    | On the |            |     |               |     |       |             |      |     |
state-of-the-artCNNmodelstoshowtheperformanceonthe
other hand, static deviation map is relatively stable, whose textclassificationtasks.
normalizedvalue∆p^Soisinitializedrandomlyandoptimized
| byback-propagation. |     |     |     |     |     |     |     | TF-MCCNN |     |     |     |     |     |     |     |
| ------------------- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
Then,thesenormalizeddeviationsareusedtocalculatethe
|            |     |      |              |        |       |         |         | As Figure | 4(a) | shows, | Multichannel |     | CNN (MCCNN) |     | is a |
| ---------- | --- | ---- | ------------ | ------ | ----- | ------- | ------- | --------- | ---- | ------ | ------------ | --- | ----------- | --- | ---- |
| deviations | by  | ∆p = | (cid:22)·∆pˆ | ◦(l;k) | where | l and k | are the |           |      |        |              |     |             |     |      |
four-layernetworkproposedby[Kim,2014]torepresenttext
lengthandwidthofinputfeaturemapand(cid:22)isthecoefficient sequences,whichonlyusedoneconvolutionlayerandpool-
| toadjustthescaleofdeviations. |     |     |     | (cid:22)issetto0:1empiricallyin |     |     |     |           |     |         |             |     |              |     |      |
| ----------------------------- | --- | --- | --- | ------------------------------- | --- | --- | --- | --------- | --- | ------- | ----------- | --- | ------------ | --- | ---- |
|                               |     |     |     |                                 |     |     |     | ing layer | for | feature | extraction. | Its | architecture | has | been |
thispaper.Thisnormalizationprocessisdesignedtokeepthe
provensimpleyetpractical.
deviationscaleindependentfromthesizeofchunks.
Forwordembedding,MCCNNexploitstwolookuptables
|     |     |     |     |     |     |     |     | tovectorizetheinputsequence. |     |     |     | Oneisinitializedbythepre- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | ------------------------- | --- | --- | --- |
2.3 NormalizedGradient
|     |     |     |     |     |     |     |     | trained | word vectors |     | that are | learned | by unsupervised |     | lan- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------------ | --- | -------- | ------- | --------------- | --- | ---- |
@L
| For the | static     | deviations, | the | gradients    | of them    |     | deter- |                |     |             |           |                            |          |           |     |
| ------- | ---------- | ----------- | --- | ------------ | ---------- | --- | ------ | -------------- | --- | ----------- | --------- | -------------------------- | -------- | --------- | --- |
|         |            |             |     |              |            | @∆p |        | guagemodelsto  |     | overcomethe |           | absenceoflarge-scalesuper- |          |           |     |
| mine    | the moving | speed       | of  | the sampling | locations. |     | How-   |                |     |             |           |                            |          |           |     |
|         |            |             |     |              |            |     |        | vised training |     | set. The    | other one | has                        | the same | size with | the |
ever,theyaredependentonotherchangeableparameterslike
former,butitisrandomlyinitialized.
the weights or inputs. So their gradient values may change For feature extraction, MCCNN first uses a convolutional
| drastically,                           | whichmakesthemagnitudeofdeviationsunpre- |     |     |     |                  |                |     |                                                    |          |         |              |     |            |     |         |
| -------------------------------------- | ---------------------------------------- | --- | --- | --- | ---------------- | -------------- | --- | -------------------------------------------------- | -------- | ------- | ------------ | --- | ---------- | --- | ------- |
|                                        |                                          |     |     |     |                  |                |     | layer with                                         | multiple | filters | (3 different |     | sizes with | 100 | filters |
| dictable.                              | Ifthedeviationsaretoosmall,              |     |     |     | themodulescannot |                |     |                                                    |          |         |              |     |            |     |         |
|                                        |                                          |     |     |     |                  |                |     | each). Eachfilterwisappliedonbothembeddingresultsx |          |         |              |     |            |     | c       |
| liveuptothepotentialastheyareexpected. |                                          |     |     |     |                  | Onthecontrary, |     |                                                    |          |         |              |     |            |     |         |
andaddeduptoemitthefeaturemap:
| the sampling |     | locations | are too | dispersed | to  | obtain useful | in- |     |     |     |     |     |     |     |     |
| ------------ | --- | --------- | ------- | --------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∑
formation. Hence, controlling the gradient of deviations is F = f(w∗x +b) (10)
c
necessary.
c
4498

ProceedingsoftheTwenty-SeventhInternationalJointConferenceonArtificialIntelligence(IJCAI-18)
Dataset Train Dev. Test Voc. Len. Class
fcfc l alayyeerr ccoonnvv22 mmaaxx ppoooolliinngg
SST-1 8544 1101 2210 18K 18 5
SST-2 6920 872 1821 16K 19 2
mmaaxx p poooolilningg kk--mmaaxx p poooolilningg ffoolldd IMDB 25000 - 25000 392K 279 2
SUBJ 8000 1000 1000 21K 23 2
ccoonnvv ccoonnvv11 ffcc llaayyeerr TREC 4907 545 500 10K 10 6
Books 1398 200 400 22K 159 2
Electronics 1398 200 400 11K 111 2
eemmbbeeddddiningg eemmbbeeddddiinngg
DVDs 1400 200 400 22K 189 2
(a) MCCNN (b) DCNN Kitchen 1400 200 400 10K 93 2
Figure4:ComparisonoftwoplainCNNsfortextclassification. (a) Table1:Statisticsofthetextclassificationdatasets.Train,Dev.and
WereplaceonlytheconvolutionallayersforTF-ConvNet. (b)We Testdenotethesizeoftrain,developmentandtestsetrespectively;
replacebothconvolutionalandpoolinglayers. Voc.:Vocabularysize;Len.:Averagesentencelength.
wheref istherectifiedlinearunitandbisthebias. Then,the
books, electronics, DVDs and kitchen. These corpora are
300 output feature maps are put into max-over-time pooling
classified according to the sentiment of positiveness or neg-
layertoforma300-dimensionfeaturemap.
ativeness. They are derived from the raw data published by
For comparison and avoiding changing the structure of [Blitzeretal.,2007].
MCCNN too much, we just replace the convolutional layer
with transformable convolution and keep the rest constant. 4.2 CompetitorMethods
ThenewlyconstructedmodelisreferredtoasTF-MCCNN.
Besidesourbaselinemodels(DCNNandMCCNN)4,wealso
TF-DCNN
compare our approaches with other state-of-the-art models,
Dynamicconvolutionalneuralnetwork(DCNN)[Kalchbren- involving NBOW, RTNT [Socher et al., 2013], PV [Le and
ner et al., 2014] is a deeper and slimmer text classification Mikolov,2014],MGNC-CNN[Zhangetal.,2016]andTree-
modelwith7layers,whichisnamedafteritsdynamick-max LSTM[Taietal.,2015]. ForLSTMandBiLSTMinTable2,
pooling. As Figure 4(b) shows, it alternates wide convolu- weusedthearchitecturesimplementedby[Jo´zefowiczetal.,
tionallayerswithpoolinglayerstocapturethehigh-levelse- 2015],whichremovethepeep-holefromthesimilararchitec-
manticfeatures. Twoconvolutionallayersusedifferentfilters tureof[Graves,2013].
of size 2 and 3 respectively. And the maximum widths of
chunksintwopoolinglayersare setto 5 and 3 respectively. 4.3 Hyper-parameters
Italsoinsertsafoldinglayerbeforethesecondpoolinglayer
tocreatetheinteractionacrossdimensions. For TF-MCCNN, as original, we use word vectors from
To transform DCNN into a TF-ConvNet, we replace all
Word2Vec5,whichistrainedbybag-of-wordsmodelon100B
the convolutional and pooling layers with the corresponding Google News corpus [Mikolov et al., 2013]. For structure,
transformable modules and keep the others same. And the TF-MCCNN uses three different lengths of 5;6;7 for ev-
newmodeliscalledTF-DCNN. ery100filters. Correspondingly, thesizesofsamplingloca-
tionsforD andS aresetto2;3;4and3;3;3respectively.
c c
Fortraining,allinvolvedparametersarerandomlyinitialized
4 Experiments
fromatruncatednormaldistributionwithzeromeanandstan-
4.1 Datasets dard deviation. And the learning rates are 10(cid:0)3;10(cid:0)4 for
the first 2 and last 1 iterations. Mini-batch is generated by
We extensively evaluate our deformable ConvNets on 9 3 3
randomly selecting 50 samples every time from the corpus.
datasets, which are collected in different domains with dif-
Training is implemented through SGD (stochastic gradient
ferentlabels. TheirstatisticsarelistedinTable1.
descend)withtheAdadeltaupdaterule[Zeiler,2012].
SST-1, SST-2 [Socher et al., 2013] and IMDB [Maas et
al.,2011]areaboutmoviereviews,whichallclassifythere- For TF-DCNN, parameters are normally initialized in [-
0.1, 0.1]. The filter widths in transformable convolutional
viewsaccordingthesentimentbutIMDBhaslongercontent
andSST-1hasmoreclasses. SUBJ2 isasubjectivitydataset, layer are 5;6 with |D c | = 2;3 and |S c | = 3;3. The shapes
ofchunksintwotransformablepoolinglayersare7×1;5×
whose task is to classify a sentence level text as being sub-
jective or objective [Pang et al., 2004]; TREC3 dataset has 1 with |D o | = 5;3;|S o | = 2;2. For training, the network
is trained with mini-batches of size 50 by back-propagation
the task of classifying a question into 6 types (the questions
areaboutlocation,person,numericinformation,etc.) [Liand and the gradient-based optimization is performed using the
Roth,2002] Adagradupdaterule[Duchietal.,2011]. Learningratesare
alsosetto10(cid:0)3;10(cid:0)4forthefirst 2 andlast 1 iterations.
Therest4datasetsareallproductreviews,whicharecom- 3 3
prised of Amazon product reviews in 4 domains, including
4We use the published results and fill in the rest with released
2http://www.cs.cornell.edu/people/pabo/movie-review-data/ codes.
3http://cogcomp.cs.illinois.edu/Data/QA/QC/ 5https://code.google.com/p/word2vec/
4499

ProceedingsoftheTwenty-SeventhInternationalJointConferenceonArtificialIntelligence(IJCAI-18)
Models SST-1 SST-2 IMDB SUBJ TREC Books Electronics DVDs Kitchen
|     | NBOW      | 42.4 80.5 | 83.6 91.3 | 88.2      | - -  | -    | -    |     |
| --- | --------- | --------- | --------- | --------- | ---- | ---- | ---- | --- |
|     | RNTN      | 45.7 85.4 | - -       | -         | - -  | -    | -    |     |
|     | PV        | 44.6 82.7 | 91.7 90.5 | 91.8      | - -  | -    | -    |     |
|     | Tree-LSTM | 50.6 86.5 | - -       | -         | - -  | -    | -    |     |
|     | LSTM      | 45.9 85.8 | 88.5 91.6 | 92.5 79.5 | 80.5 | 81.7 | 78.0 |     |
|     | BiLSTM    | 46.4 86.1 | 89.9 90.1 | 92.7 81.0 | 78.5 | 80.5 | 81.2 |     |
|     | MGNC-CNN  | 48.7 88.3 | - 94.1    | -         | - -  | -    | -    |     |
|     | DCNN      | 48.5 86.8 | 88.6 93.6 | 93.0 79.7 | 80.4 | 81.6 | 77.8 |     |
|     | MCCNN     | 47.4 88.1 | 89.7 93.2 | 92.2 79.9 | 77.0 | 81.2 | 77.7 |     |
|     | TF-DCNN   | 48.8 88.1 | 89.9 95.0 | 93.2 82.0 | 81.7 | 81.3 | 81.2 |     |
|     | TF-MCCNN  | 49.1 88.5 | 90.3 94.2 | 93.5 81.5 | 81.1 | 81.6 | 81.4 |     |
Table2:Accuraciesofourmodelsandcomparison.
-3   --32      --21      - 10        01        1 2         23     3
-3   --32      --21      - 10        01        1 2         23     3
1.00
|      |     |     |     | ConvC1o_n1v1_1 | ConvC1o_n2v1_2 |                |     | ConvC1o_n2v1_2 |
| ---- | --- | --- | --- | -------------- | -------------- | -------------- | --- | -------------- |
| 0.75 |     |     |     |                |                | ConvC1o_n1v1_1 |     |                |
0.50 -3   -2-3     - 1-2       0-1       1 0       2 1         32     - 33    -2-3     - 1-2       0-1       1 0       2 1         32      3  -3   -2-3     - 1-2       0-1       1 0       2 1         32     - 33    -2-3     - 1-2       0-1       1 0       2 1         32      3
|      |     |     |     | ConvC2o_n1v2_1 | ConvC2o_n2v2_2 |                |     | ConvC1o_n4v1_4 |
| ---- | --- | --- | --- | -------------- | -------------- | -------------- | --- | -------------- |
| 0.25 |     |     |     |                |                | ConvC1o_n3v1_3 |     |                |
0.00
aa  scresecrneepnlaplyay momreor eingeinngeioniuoussllyy ccoonsntrsutcrtuedctedth athnan      '“'      mmeemmeentnoto     "” -3   -2-3     - 1-2       0-1       1 0       2 1         32     - 33    -2-3     - 1-2       0-1       1 0       2 1         32      3  -3   -2-3     - 1-2       0-1       1 0       2 1         32      -33    -2-3     - 1-2       0-1       1 0       2 1         32      3
(a) ascreenplaymoreingeniouslyconstructedthan“memento” (a) Learnedstaticsamplingloca-(b) Learnedstaticsamplingloca-
|      |     |      |     | tionsinTF-DCNN |                | tionsinTF-MCCNN |     |     |
| ---- | --- | ---- | --- | -------------- | -------------- | --------------- | --- | --- |
| 1.00 |     | 1.00 |     |                | ConvC1_o1nv1_1 |                 |     |     |
Figure6:Visualizationofthestaticpartoffilters.Redpointsdenote
0.75
0.75 the learned samp-3 l  i  n-2- 3 g      -  1- 2l   o      0 -c 1    a     1 t  0 i    o    2   1n       s    3 2   o    f 3 the filters. (a) Convi j denotes
0.50
0.50 theshapeofj-thfilteCroinnvC2_oi1n-vt2h_1transformableconvolutionallayer. (b)
0.25
| 0.25 |     |      |     | FourrandomlyselectedfiltersinTF-MCCNN. |                                                                                              |     |     |     |
| ---- | --- | ---- | --- | -------------------------------------- | -------------------------------------------------------------------------------------------- | --- | --- | --- |
|      |     | 0.00 |     |                                        | -3     -2- 3       -  1- 2         0 - 1         1   0         2   1           3 2        3  |     |     |     |
0.00
| .·.·.·         ,       bbuutt       it       i sis   effective    |     | .·.·.·   but      itit        ’s     a afantastic mmoovviiee |     |     |     |     |     |     |
| ----------------------------------------------------------------- | --- | ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- |
| (b) s1m0ne’ssatireisnotsubtle,(c)                                 |     | it’stoughtowatch,butit’sa                                    |     |     |     |     |     |     |
4.5 Visualization
| butitiseffective                                |                        | fantasticmovie. |                      |                                              |                                |                   |     |          |
| ----------------------------------------------- | ---------------------- | --------------- | -------------------- | -------------------------------------------- | ------------------------------ | ----------------- | --- | -------- |
|                                                 |                        |                 |                      | To intuitively                               | understand how                 | the transformable |     | ConvNets |
|                                                 |                        |                 |                      | adapttothefeatures.                          | Wedesignanexperimenttoshowwhat |                   |     |          |
| Figure5:Visualizationofthedynamicpartoffilters. |                        |                 | Bluecolumns          |                                              |                                |                   |     |          |
|                                                 |                        |                 |                      | patternsthemodelscaptureandhowtheytransform. |                                |                   |     | Wefirst  |
| represent                                       | the sampling locations | of the filter   | that has the biggest |                                              |                                |                   |     |          |
valueofactivation.Andredlinesshowtheactivationvaluesoftrans- randomlyextractasentencefromthetestsetofSST-2dataset
formationconvolution. and feed it into TF-DCNN. Then, for the dynamic part of
filtersinthefirsttransformableconvolutionlayer,wevisual-
|     |     |     |     | ize the sampling | location of | the biggest | activation. | We also |
| --- | --- | --- | --- | ---------------- | ----------- | ----------- | ----------- | ------- |
4.4 Performance illustratealltheactivationsgeneratedbytransformationcon-
volution.Figure5visuallyshowstheeffectoftransformation
| Table | 2 reports the performance | of our approaches | against |     |     |     |     |     |
| ----- | ------------------------- | ----------------- | ------- | --- | --- | --- | --- | --- |
convolutiononthreedifferentsentences.Asthebluecolumns
other models. We can see that TF-ConvNets obviously showin5(a),thefilterskipsawordandcapturesthekeystruc-
improve the performance over their baseline models (MC- ture of “more ingeniously ... than”, which is crucial to the
CNN and DCNN), which demonstrates the effectiveness of prediction. Itisalmostimpossibleforaplainfiltertodirectly
transformable modules. Compared with other models, TF- capturethiskindoffeatures,whichgenerallyhavetobecom-
ConvNetsgainthebestperformanceon6tasksandshowvery
|     |     |     |     | posedinhigh-levellayers. |     | Besides,wecanhaveanobvious |     |     |
| --- | --- | --- | --- | ------------------------ | --- | -------------------------- | --- | --- |
competitiveresultsontheothers.
observationfromFigure5(b)and5(c)thatthefiltercanadapt
AlthoughTree-SLTMoutperformsourmodelsonSST-1,it to the transformation of features. From two sentences, the
obtains the external topological information via parser trees filteractivelylearnsthedynamicshapeofthefeatureandrec-
of sentences. Our models do not obtain the best score on ognizesitsnegationstructure“but... [Adj.]”,eventheyhave
IMDBandDVDs,asthesequencesinthemaremuchlonger differentnumbersformiddlewords.
than other datasets, which requires stronger capability for For static deviations, their values are learned via back-
long-termdependencylearning. Inthispaper,wemainlyfo- propagation and finally reach constringency. Figure 6 illus-
cus on the effects of transformation mechanism on text rep- tratesthefinalshapeslearnedbystaticpartoffilters.Fromthe
resentation, but our approach can be also applied on other extensiveexperiments,wefindaregularpatternthatthefilters
modelsdesignedforlongsequencetoproducebetterperfor- tendtodilateinlowlayerbutaremoresimilartoconventional
mances. convolutioninhighlayer. Ourconjectureisthatfiltersinlow
4500

ProceedingsoftheTwenty-SeventhInternationalJointConferenceonArtificialIntelligence(IJCAI-18)
Modules Books Elec. DVDs Kitchen Avg. variance of features in the datasets or using consistent fea-
|      |     |      |      |      |      |     |      | tures, rather | than | the inner | problem. | Some | recent | works at- |
| ---- | --- | ---- | ---- | ---- | ---- | --- | ---- | ------------- | ---- | --------- | -------- | ---- | ------ | --------- |
| DCNN |     | 79.7 | 80.4 | 81.7 | 77.8 |     | 79.9 |               |      |           |          |      |        |           |
tempttoaugmenttheadaptabilityofmodelsbyrevisingsome
| +TF-conv    |     | 81.7 | 81.3 | 81.3 | 81.1 |     | 81.4 |                                             |     |      |                  |     |             |           |
| ----------- | --- | ---- | ---- | ---- | ---- | --- | ---- | ------------------------------------------- | --- | ---- | ---------------- | --- | ----------- | --------- |
|             |     |      |      |      |      |     |      | basic structures,                           |     | such | as convolutional |     | and pooling | layers.   |
| +TF-pooling |     | 81.1 | 80.9 | 81.3 | 80.5 |     | 81.0 |                                             |     |      |                  |     |             |           |
|             |     |      |      |      |      |     |      | Theseworkscanbeclassifiedintotwocategories: |     |      |                  |     |             | semi-end- |
| +Both       |     | 82.0 | 81.7 | 81.3 | 81.2 |     | 81.6 |                                             |     |      |                  |     |             |           |
to-endmodelsandfull-end-to-endmodels.
| DCNN(0,0) |     | 79.7 | 80.4 | 81.7 | 77.8 |     | 79.9 |            |     |        |                 |      |           |         |
| --------- | --- | ---- | ---- | ---- | ---- | --- | ---- | ---------- | --- | ------ | --------------- | ---- | --------- | ------- |
|           |     |      |      |      |      |     |      | For models |     | in the | first category, | they | generally | reshape |
DCNN(2,0) 81.6 80.9 81.0 81.0 81.1 some layers with handcrafted patterns. For input, Spatial
DCNN(0,2) 80.3 80.8 79.9 79.6 80.2 Transform Network (STN) [Jaderberg et al., 2015] tries to
DCNN(2,2) 81.7 81.3 81.3 81.1 81.4 learn the transform parameters to warp the inputs, such as
|          |     |      |      |      |      |     |      | affine transformation. |     |     | For convolutional |     | layer, | Atrous Con- |
| -------- | --- | ---- | ---- | ---- | ---- | --- | ---- | ---------------------- | --- | --- | ----------------- | --- | ------ | ----------- |
| MCCNN(0) |     | 79.9 | 77.0 | 81.2 | 77.7 |     | 79.0 |                        |     |     |                   |     |        |             |
volution[Holschneideretal.,1988]improvestheconvolution
| MCCNN(10) |     | 81.2 | 80.6 | 81.6 | 80.9 |     | 81.1 |               |        |          |           |     |          |               |
| --------- | --- | ---- | ---- | ---- | ---- | --- | ---- | ------------- | ------ | -------- | --------- | --- | -------- | ------------- |
|           |     |      |      |      |      |     |      | by scattering | filter | sampling | locations |     | with the | same strides, |
| MCCNN(20) |     | 81.5 | 81.1 | 81.6 | 81.4 |     | 81.4 |               |        |          |           |     |          |               |
whichincreasesthereceptivefieldofthefilterswiththesame
|          |                |     |               |     |           |      |      | parameter | complexity. |                | For pooling | layer, | Receptive     | Field   |
| -------- | -------------- | --- | ------------- | --- | --------- | ---- | ---- | --------- | ----------- | -------------- | ----------- | ------ | ------------- | ------- |
| Table 3: | Ablation study | of  | transformable |     | ConvNets. | Avg. | col- |           |             |                |             |        |               |         |
|          |                |     |               |     |           |      |      | Learning  | [Jia        | et al., 2012], | following   |        | the [Lazebnik | et al., |
umnreportstheaverageaccuraciesof4datasetsfromamazonprod-
2006],learnsasubsetofpoolingregionsviatheideaofover-
| uctreviews. | ThetuplesfollowingDCNNrepresentthenumbersof |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
usedtransformableconvolutioninthefirstandsecondconvolutional completefromalargenumberofcandidates.
layer.ThenumbersfollowingMCCNNdenotethenumbersoftrans- However,intheseworksthetransformationinformationis
formableconvolutionusedinMCCNN. knownaspriorandthepatternsfortransformationarelimited.
|     |     |     |     |     |     |     |     | They cannot  | deal | with       | the intricate | transformations |          | that are     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | ---------- | ------------- | --------------- | -------- | ------------ |
|     |     |     |     |     |     |     |     | out of human |      | knowledge. | Moreover,     |                 | they are | designed for |
layersexpandtocapturelongerfeatures,whilehigherlayers
|     |     |     |     |     |     |     |     | computer | vision | and are | not suitable | for | text representation |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------ | ------- | ------------ | --- | ------------------- | --- |
focusonbuildingconnectionsbetweenclosefeatures.
tasks,whichhavedifferentdimensionsoftransformation.
|     |     |     |     |     |     |     |     | For the | second | category, | models | learn | the sampling | loca- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------ | --------- | ------ | ----- | ------------ | ----- |
5 AblationStudy
tionautonomouslyandthetransformationisnotrestrictedby
|                  |     |         |      |                |     |          |     | manuallymadepatterns.          |     |     | Thatmeansthiskindofmodelsare |                     |     |     |
| ---------------- | --- | ------- | ---- | -------------- | --- | -------- | --- | ------------------------------ | --- | --- | ---------------------------- | ------------------- | --- | --- |
| In this section, | we  | conduct | more | investigations |     | to study | the |                                |     |     |                              |                     |     |     |
|                  |     |         |      |                |     |          |     | moregeneralandfullyend-to-end, |     |     |                              | whichcanadapttomore |     |     |
independenteffectofeachmoduleandtheinfluenceoftrans-
formableconvolutionnumber. intricatetransformations. Tobespecific,ActiveConvolution
[JeonandKim,2017]learnstheshapesoffiltersforconvolu-
EffectofEachModule tion,whichcanadjustitselfaccordingtheglobaldatasetsvia
Toprovetheeffectivenessofbothmodules, wedosomeab- back-propagation. However, the shapes of filters are static
andarelearnedpertaskorpertraining,lackingreactivityfor
lationexperimentsonTF-DCNNsinceitusesbothmodules.
currentinputorlocalfeatures.
| Table 3 | shows the | performance |     | of variants | on  | 4 product | re- |     |     |     |     |     |     |     |
| ------- | --------- | ----------- | --- | ----------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
viewdatasets. Whentransformableconvolutionisappliedon Another end-to-end model Deformable CNN [Dai et al.,
the network, an average performance improvement of 1.5% 2017]attemptstomodelthetransformationthroughlearning
is obtained. With transformable pooling only, it performs a theoffsetsforconvolutionfiltersandpoolingregions. Italso
| boostofaverage1.1%. |     | Thisdemonstratesthatbothmodules |     |     |     |     |     |                                        |     |     |     |     |     |              |
| ------------------- | --- | ------------------------------- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- | --- | ------------ |
|                     |     |                                 |     |     |     |     |     | providesfreedomtoshapesforbothmodules, |     |     |     |     |     | whiletheoff- |
makeacontributiontothepromisingscores. sets drastically vary with the current input and sampling lo-
|     |     |     |     |     |     |     |     | cation. Thismakesithighlyfitthecurrentfieldlosingoutline |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
NumberofTransformableConvolution
andconsistentpatternsfortheintegraltask.
OnbothConvNets,weimplementaseriesofincrementalex-
|                          |                       |               |                            |                  |       |          |     | In this     | paper,  | our modules       |      | overcome | the          | drawbacks of |
| ------------------------ | --------------------- | ------------- | -------------------------- | ---------------- | ----- | -------- | --- | ----------- | ------- | ----------------- | ---- | -------- | ------------ | ------------ |
| periments                | to study              | the influence |                            | of transformable |       | convolu- |     |             |         |                   |      |          |              |              |
|                          |                       |               |                            |                  |       |          |     | above works | and     | consider          | both | current  | and global   | transfor-    |
| tionnumberonperformance. |                       |               | TheresultsreportedinTable3 |                  |       |          |     |             |         |                   |      |          |              |              |
|                          |                       |               |                            |                  |       |          |     | mations.    | It is   | noteworthy        | that | our work | successfully | intro-       |
| indicate                 | that 1) Transformable |               |                            | convolution      | shows | more     | im- |             |         |                   |      |          |              |              |
|                          |                       |               |                            |                  |       |          |     | duces the   | concept | of transformation |      | into     | language     | field and    |
provement in lower layers. Because it is mainly designed enhances the flexibility and adaptability of CNN models for
| for feature | extraction, | while | it  | also works | in high | levels | for |     |     |     |     |     |     |     |
| ----------- | ----------- | ----- | --- | ---------- | ------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
textrepresentation.
| featurefusion. | ThisconcurswiththephenomenoninFigure |          |             |         |               |             |         |                |     |              |     |            |          |      |
| -------------- | ------------------------------------ | -------- | ----------- | ------- | ------------- | ----------- | ------- | -------------- | --- | ------------ | --- | ---------- | -------- | ---- |
| 6(a) that      | filter shapes                        | change   | more        | in      | lower layers. |             | 2) Per- |                |     |              |     |            |          |      |
| formance       | steadily                             | improves | when        | more    | transformable |             | con-    | 7 Conclusion   |     |              |     |            |          |      |
| volution       | layers are                           | involved | until       | a point | that          | improvement |         |                |     |              |     |            |          |      |
|                |                                      |          |             |         |               |             |         | In this paper, |     | by combining | two | mainstream | methods, | pro- |
| saturates.     | Specifically,                        | the      | improvement |         | saturates     | when        | 20      |                |     |              |     |            |          |      |
transformablelayersareusedinMCCNN. posed modules enhance the flexibility of structure for CNN
|     |     |     |     |     |     |     |     | andaugmenttheadaptabilityfortransformablefeatures. |     |     |     |     |     | Our |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
approachovercomesthedrawbacksofn-grambasedmodels
6 RelatedWork
fortextclassificationandshowsanabilitytocapturetheover-
CNN inherently suffers from the limitation to model the length or non-consecutive features. Extensive tests and vi-
transformationofobjectsbecauseofitsfixedgeometricstruc- sualizationexperimentsdemonstratetheeffectivenessofour
ture.Traditionally,mostoftheworksfocusoncoveringmore modulesforfeaturetransformation.
4501

ProceedingsoftheTwenty-SeventhInternationalJointConferenceonArtificialIntelligence(IJCAI-18)
| References |     |     |     |     |     |     |     | (CVPR2006),17-22June2006,NewYork,NY,USA,pages |     |     |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
2169–2178,2006.
| [Bayetal.,2006] |     | Herbert | Bay, | Tinne | Tuytelaars, |     | and Luc |     |     |     |     |     |     |     |     |
| --------------- | --- | ------- | ---- | ----- | ----------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
J.VanGool. SURF:speededuprobustfeatures. InCom- [LeandMikolov,2014] Quoc V. Le and Tomas Mikolov.
| puter | Vision | - ECCV | 2006, | 9th European |     | Conference | on  |             |                 |     |     |              |     |            |     |
| ----- | ------ | ------ | ----- | ------------ | --- | ---------- | --- | ----------- | --------------- | --- | --- | ------------ | --- | ---------- | --- |
|       |        |        |       |              |     |            |     | Distributed | representations |     |     | of sentences | and | documents. |     |
Computer Vision, Graz, Austria, May 7-13, 2006, Pro- ComputerScience,4:1188–1196,2014.
ceedings,PartI,pages404–417,2006.
|     |     |     |     |     |     |     |     | [LiandRoth,2002] |     | XinLiandDanRoth.Learningquestion |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | -------------------------------- | --- | --- | --- | --- | --- |
[Blitzeretal.,2007] John Blitzer, Mark Dredze, and Fer- classifiers. Coling,12(24):556–562,2002.
| nandoPereira. |        | Biographies, |     | bollywood, |           | boom-boxesand   |       |                                 |     |          |      |          |      |                |     |
| ------------- | ------ | ------------ | --- | ---------- | --------- | --------------- | ----- | ------------------------------- | --- | -------- | ---- | -------- | ---- | -------------- | --- |
|               |        |              |     |            |           |                 |       | [Liuetal.,2015]                 |     | Xiaodong | Liu, | Jianfeng | Gao, | Xiaodong       |     |
| blenders:     | Domain | adaptation   |     | for        | sentiment | classification. |       |                                 |     |          |      |          |      |                |     |
|               |        |              |     |            |           |                 |       | He,LiDeng,KevinDuh,andYeYiWang. |     |          |      |          |      | Representation |     |
| In ACL        | 2007,  | Proceedings  |     | of ACL,    | June      | 23-30,          | 2007, |                                 |     |          |      |          |      |                |     |
learningusingmulti-taskdeepneuralnetworksforseman-
Prague,CzechRepublic,pages187–205,2007.
|                 |     |        |      |        |     |       |        | tic classification |     | and | information | retrieval. |     | In In | NAACL |
| --------------- | --- | ------ | ---- | ------ | --- | ----- | ------ | ------------------ | --- | --- | ----------- | ---------- | --- | ----- | ----- |
| [Daietal.,2017] |     | Jifeng | Dai, | Haozhi | Qi, | Yuwen | Xiong, |                    |     |     |             |            |     |       |       |
HLT,pages912–921,2015.
| Yi Li,                         | Guodong | Zhang, | Han | Hu, | and                  | Yichen | Wei. De- |             |              |     |     |                            |     |     |     |
| ------------------------------ | ------- | ------ | --- | --- | -------------------- | ------ | -------- | ----------- | ------------ | --- | --- | -------------------------- | --- | --- | --- |
|                                |         |        |     |     |                      |        |          | [Lowe,1999] | DavidG.Lowe. |     |     | Objectrecognitionfromlocal |     |     |     |
| formableconvolutionalnetworks. |         |        |     |     | CoRR,abs/1703.06211, |        |          |             |              |     |     |                            |     |     |     |
scale-invariantfeatures.InICCV,pages1150–1157,1999.
2017.
|                               |                                           |      |        |      |                |     |           | [Maasetal.,2011] |          | Andrew | L       | Maas, Raymond |         | E Daly,     | Pe- |
| ----------------------------- | ----------------------------------------- | ---- | ------ | ---- | -------------- | --- | --------- | ---------------- | -------- | ------ | ------- | ------------- | ------- | ----------- | --- |
| [Duchietal.,2011]             |                                           | John | Duchi, | Elad | Hazan,         |     | and Yoram |                  |          |        |         |               |         |             |     |
|                               |                                           |      |        |      |                |     |           | ter T Pham,      | Dan      | Huang, | Andrew  | Y             | Ng, and | Christopher |     |
| Singer.                       | AdaptiveSubgradientMethodsforOnlineLearn- |      |        |      |                |     |           |                  |          |        |         |               |         |             |     |
|                               |                                           |      |        |      |                |     |           | Potts.           | Learning | word   | vectors | for sentiment |         | analysis.   | In  |
| ingandStochasticOptimization. |                                           |      |        |      | JMLR.org,2011. |     |           |                  |          |        |         |               |         |             |     |
MeetingoftheAssociationforComputationalLinguistics,
| [Graves,2013] |     | AlexGraves. |     | Generatingsequenceswithre- |     |     |     |     |     |     |     |     |     |     |     |
| ------------- | --- | ----------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
pages142–150,2011.
| currentneuralnetworks. |     |     | CoRR,abs/1308.0850,2013. |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------------------- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[Mikolovetal.,2013]
|                          |     |     |     |               |     |     |           |            |          | Tomas |             | Mikolov, | Ilya        | Sutskever, | Kai  |
| ------------------------ | --- | --- | --- | ------------- | --- | --- | --------- | ---------- | -------- | ----- | ----------- | -------- | ----------- | ---------- | ---- |
| [Holschneideretal.,1988] |     |     | M.  | Holschneider, |     | R.  | Kronland- |            |          |       |             |          |             |            |      |
|                          |     |     |     |               |     |     |           | Chen, Greg | Corrado, |       | and Jeffrey | Dean.    | Distributed |            | rep- |
Martinet, J. Morlet, and Ph. Tchamitchian. A real-time resentations of words and phrases and their composition-
algorithmfor signal analysis with the help of the wavelet ality.AdvancesinNeuralInformationProcessingSystems,
| transform. | pages286–297,1988. |     |     |     |     |     |     | 26:3111–3119,2013. |     |     |     |     |     |     |     |
| ---------- | ------------------ | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- |
[Jaderbergetal.,2015] Max Jaderberg, Karen Simonyan, [Pangetal.,2004] Pang, Bo, Lee, andLillian. Asentimen-
| Andrew      | Zisserman, |           | and | Koray    | Kavukcuoglu. |        | Spatial  |                               |     |           |          |       |                   |     |      |
| ----------- | ---------- | --------- | --- | -------- | ------------ | ------ | -------- | ----------------------------- | --- | --------- | -------- | ----- | ----------------- | --- | ---- |
|             |            |           |     |          |              |        |          | tal education:                |     | sentiment | analysis | using | subjectivity      |     | sum- |
| transformer |            | networks. | In  | Advances | in           | Neural | Informa- |                               |     |           |          |       |                   |     |      |
|             |            |           |     |          |              |        |          | marizationbasedonminimumcuts. |     |           |          |       | ProceedingsofACL, |     |      |
tionProcessingSystems,December7-12,2015,Montreal,
pages271–278,2004.
Quebec,Canada,pages2017–2025,2015.
|                   |     |     |       |          |       |      |        | [Rubleeetal.,2011] |     | Ethan | Rublee,     | Vincent |     | Rabaud,   | Kurt   |
| ----------------- | --- | --- | ----- | -------- | ----- | ---- | ------ | ------------------ | --- | ----- | ----------- | ------- | --- | --------- | ------ |
| [JeonandKim,2017] |     |     | Yunho | Jeon and | Junmo | Kim. | Active |                    |     |       |             |         |     |           |        |
|                   |     |     |       |          |       |      |        | Konolige,          | and | Gary  | R. Bradski. | ORB:    | an  | efficient | alter- |
convolution: Learningtheshapeofconvolutionforimage native to SIFT or SURF. In IEEE International Confer-
| classification. |     | CoRR,abs/1703.09076,2017. |     |     |     |     |     |         |          |         |      |       |            |     |        |
| --------------- | --- | ------------------------- | --- | --- | --- | --- | --- | ------- | -------- | ------- | ---- | ----- | ---------- | --- | ------ |
|                 |     |                           |     |     |     |     |     | ence on | Computer | Vision, | ICCV | 2011, | Barcelona, |     | Spain, |
[Jiaetal.,2012] Yangqing Jia, Chang Huang, and Trevor November6-13,2011,pages2564–2571,2011.
| Darrell.                   | Beyond | spatial | pyramids: |     | Receptive          |     | field learn- |                  |       |        |           |          |     |                 |          |
| -------------------------- | ------ | ------- | --------- | --- | ------------------ | --- | ------------ | ---------------- | ----- | ------ | --------- | -------- | --- | --------------- | -------- |
|                            |        |         |           |     |                    |     |              | [Shenetal.,2014] |       | Yelong | Shen,     | Xiaodong |     | He,             | Jianfeng |
| ingforpooledimagefeatures. |        |         |           | In  | 2012IEEEConference |     |              |                  |       |        |           |          |     |                 |          |
|                            |        |         |           |     |                    |     |              | Gao, Li          | Deng, | and    | Gre´goire | Mesnil.  | A   | latent semantic |          |
onComputerVisionandPatternRecognition,Providence,
|     |     |     |     |     |     |     |     | model | with convolutional-pooling |     |     | structure |     | for informa- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | -------------------------- | --- | --- | --------- | --- | ------------ | --- |
RI,USA,June16-21,2012,pages3370–3377,2012. tion retrieval. In Proceedings of CIKM 2014, Shanghai,
[Jo´zefowiczetal.,2015] Rafal Jo´zefowicz, Wojciech China,November3-7,2014,pages101–110,2014.
Zaremba,andIlyaSutskever. Anempiricalexplorationof [Socheretal.,2013] R. Socher, A. Perelygin, J. Y. Wu,
| recurrentnetworkarchitectures. |        |         |      | InProceedingsof,ICML |       |     |            |            |     |             |     |           |     |           |     |
| ------------------------------ | ------ | ------- | ---- | -------------------- | ----- | --- | ---------- | ---------- | --- | ----------- | --- | --------- | --- | --------- | --- |
|                                |        |         |      |                      |       |     |            | J. Chuang, | C.  | D. Manning, |     | A. Y. Ng, | and | C. Potts. | Re- |
| 2015,                          | Lille, | France, | 6-11 | July 2015,           | pages |     | 2342–2350, |            |     |             |     |           |     |           |     |
cursivedeepmodelsforsemanticcompositionalityovera
2015.
|                               |            |          |            |               |                    |     |        | sentimenttreebank. |                        |          | 2013. |                 |          |           |      |
| ----------------------------- | ---------- | -------- | ---------- | ------------- | ------------------ | --- | ------ | ------------------ | ---------------------- | -------- | ----- | --------------- | -------- | --------- | ---- |
| [Kalchbrenneretal.,2014]      |            |          | Nal        | Kalchbrenner, |                    |     | Edward |                    |                        |          |       |                 |          |           |      |
|                               |            |          |            |               |                    |     |        | [Taietal.,2015]    |                        | Kai      | Sheng | Tai, Richard    |          | Socher,   | and  |
| Grefenstette,                 |            | and Phil | Blunsom.   |               | A convolutional    |     | neural |                    |                        |          |       |                 |          |           |      |
|                               |            |          |            |               |                    |     |        | Christopher        | D.                     | Manning. |       | Improved        | semantic | represen- |      |
| networkformodellingsentences. |            |          |            |               | InProceedingsofACL |     |        |                    |                        |          |       |                 |          |           |      |
|                               |            |          |            |               |                    |     |        | tations            | from tree-structured   |          |       | long short-term |          | memory    | net- |
| 2014,                         | June22-27, | 2014,    | Baltimore, |               | MD,USA,Volume1:    |     |        |                    |                        |          |       |                 |          |           |      |
|                               |            |          |            |               |                    |     |        | works.             | ComputerScience,5(1):: |          |       | 36.,2015.       |          |           |      |
LongPapers,pages655–665,2014.
[Zeiler,2012]
|                         |      |      |                  |     |        |            |     |                     | Matthew |     | D. Zeiler.            | Adadelta: |     | An  | adaptive |
| ----------------------- | ---- | ---- | ---------------- | --- | ------ | ---------- | --- | ------------------- | ------- | --- | --------------------- | --------- | --- | --- | -------- |
| [Kim,2014]              | Yoon | Kim. | Convolutional    |     | neural | networks   | for |                     |         |     |                       |           |     |     |          |
|                         |      |      |                  |     |        |            |     | learningratemethod. |         |     | ComputerScience,2012. |           |     |     |          |
| sentenceclassification. |      |      | InProceedingsof, |     |        | EMNLP2014, |     |                     |         |     |                       |           |     |     |          |
October25-29,2014,Doha,Qatar,AmeetingofSIGDAT, [Zhangetal.,2016] YeZhang,StephenRoller,andByronC.
a Special Interest Group of the ACL, pages 1746–1751, Wallace. MGNC-CNN: A simple approach to exploiting
|     |     |     |     |     |     |     |     | multiple | word | embeddings |     | for sentence | classification. |     | In  |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---- | ---------- | --- | ------------ | --------------- | --- | --- |
2014.
|                      |     |     |                                  |     |     |     |     | NAACL | HLT 2016, | San | Diego | California, |     | USA, June | 12- |
| -------------------- | --- | --- | -------------------------------- | --- | --- | --- | --- | ----- | --------- | --- | ----- | ----------- | --- | --------- | --- |
| [Lazebniketal.,2006] |     |     | SvetlanaLazebnik,CordeliaSchmid, |     |     |     |     |       |           |     |       |             |     |           |     |
17,2016,pages1522–1527,2016.
| and Jean                                         | Ponce. | Beyond |     | bags of | features: | Spatial | pyra- |     |     |     |     |     |     |     |     |
| ------------------------------------------------ | ------ | ------ | --- | ------- | --------- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
| midmatchingforrecognizingnaturalscenecategories. |        |        |     |         |           |         | In    |     |     |     |     |     |     |     |     |
4502