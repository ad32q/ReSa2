mkdir data
cd data

# Create directory structure
mkdir msmarco_passage
mkdir msmarco_passage/raw
cd msmarco_passage/raw

# Rename query file
mv dev.query.txt queries.dev.small.tsv

# Join the paragraph and title as the final corpus 
# (different from the official collections.tsv, which does not contain the title field)
join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv

# Process official train triples (ids) file for (initial) negatives
gunzip qidpidtriples.train.full.2.tsv.gz
awk -v RS='\r\n' '$1==last {printf ",%s",$3; next} NR>1 {print "";} {last=$1; printf "%s\t%s",$1,$3;} END{print "";}' qidpidtriples.train.full.2.tsv > train.negatives.tsv
