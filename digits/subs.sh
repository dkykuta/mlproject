FILES=`ls $1 -1`

for f in $FILES; do
	extension="${f##*.}"
	newname=`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 16`.$extension
	mv $1/$f $1/$newname
done
