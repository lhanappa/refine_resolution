#setup env and check pre request

#HiCSR
HiCSR=env_hicsr
ENVS=$(conda env list | awk '{print $1}')

echo $ENVS
if [[ $ENVS = *"${HiCSR}"* ]]; then
   echo ${HiCSR} exist
   echo updating
   conda env update  --file ./environment_hicsr.yaml  --prune
else 
   echo "Error: Please provide a valid virtual environment. For a list of valid virtual environment, please see 'conda env list' "
   echo "Creating env $HiCSR from ./HiCSR/environment.yaml"
   echo conda env create -f ./environment_hicsr.yaml
   conda env create --file ./environment_hicsr.yaml
   exit
fi;


#deephic
deephic=env_deephic
ENVS=$(conda env list | awk '{print $1}')
echo $ENVS
if [[ $ENVS = *"${deephic}"* ]]; then
   echo ${deephic} exist
   echo updating
   conda env update  --file ./environment_deephic.yaml  --prune
else 
   echo "Error: Please provide a valid virtual environment. For a list of valid virtual environment, please see 'conda env list' "
   echo "Creating env $deephic from ./HiCSR/environment.yaml"
   echo conda env create -f ./environment_deephic.yaml
   conda env create --file ./environment.yaml
   exit
fi;

conda env list