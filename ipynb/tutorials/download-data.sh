# Script to download 100 16S V4 test studies.
# Note: these studies were subset from these commands
# >export ctx=Deblur_2021.09-Illumina-16S-V4-100nt-50b3a2
# >redbiom search metadata EMP | redbiom fetch sample-metadata --context $ctx --output bulk-qiita-emp.txt
# >redbiom search metadata feces | redbiom fetch sample-metadata --context $ctx --output bulk-qiita-feces.txt --all-columns
# for only cross-sectional data (and data without blooms etc..).
## make list of studies and set context type for red biome (seperated by EMP env. samples vs. Fecal)
export ctx=Deblur_2021.09-Illumina-16S-V4-100nt-50b3a2
declare -a empstudies=("1665" "2182" "1799" "1031" "1289" "1034" "1035" "1036" "910" "1041" "659" "662" "1692" "1694" "2080" "804" "933" "1702" "1064" "809" "1197" "1198" "10798" "1714" "10933" "10934" "1717" "1721" "958" "963" "10308" "1222" "1734" "1736" "846" "1235" "10323" "1627" "861" "1632" "10353" "632" "638")
# get all samples to download EMP
for i in "${empstudies[@]}"
do
   echo $i
   # download
   redbiom search metadata "EMP & qiita_study_id where qiita_study_id == $i" >> bulk-qiita-samples.txt
done
# get all samples to download fecal
declare -a fecalstudies=("13338" "10527" "11043" "11815" "12842" "13870" "11326" "13631" "10560" "11841" "833" "10563" "13639" "10567" "10315" "850" "13911" "2136" "15193" "10330" "10333" "11360" "865" "10342" "11110" "10856" "11625" "11888" "14966" "12663" "13433" "11129" "10378" "11402" "1939" "10902" "10904" "11673" "15006" "1696" "1700" "1189" "11176" "15016" "1448" "10178" "2248" "11210" "10959" "2260" "13781" "14812" "10461" "2014" "15071" "10724" "1517") 
for i in "${fecalstudies[@]}"
do
   echo $i
   # download
   redbiom search metadata "feces where qiita_study_id == $i" >> bulk-qiita-samples.txt
done
# download all data
redbiom fetch sample-metadata --from bulk-qiita-samples.txt --context $ctx --output bulk-qiita.txt --force-category "host_subject_id"
redbiom fetch samples --from bulk-qiita-samples.txt --context $ctx --output bulk-qiita.biom
