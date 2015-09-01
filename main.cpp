#include "svm_scale.h"
#include "svm_train.h"
#include "svm_predict.h"
#include "svm_crossval.h"
#include "sstream"
#include "vector"
#include "string"
using namespace std;

/** Control LibSVM
    Mode:
    [1] command line
        One command per run
    [2] console=
        Vary commands per run
*/

void exit_with_help()
{
	printf(
    "Wrong console !\n"
    "Usage: [option] <based on option>\n"
    "option:\n"
    "svm-scale: scale\n"
    "svm-train: train\n"
    "svm-predict: predict\n"
    "svm-crossval: Cross-validation (in multi case)\n\n"
    );
    exit(1);
}

#define scaleComment "svm-scale"
#define trainComment "svm-train"
#define predictComment "svm-predict"
#define crossvalComment "svm-crossval"
#define minimumInputInstant 3

string programName = "classifier_v1";
string scaleFile;
bool inConsole = false;

vector<string> splitString(string s, char delim);

int main(int argc,char **argv)
{
    string cmd;
    vector<string>cmdSplit;
    do
    {
        //console input
        while(argc<2)
        {
            inConsole = true;
            cout<<"[New comment]>> ";
            fflush(stdin);
            getline(cin,cmd);
            cmdSplit=splitString(cmd,' ');
            argc=cmdSplit.size() + 1;
            if(argc<minimumInputInstant)exit_with_help();

            argv=new char*[argc];
            argv[0]=new char[programName.length()];
            strcpy(argv[0],programName.c_str());
            for(int i=1;i<argc;i++)
            {
                stringstream ss(cmdSplit[i-1]);
                argv[i]=new char[1];
                ss>>argv[i];
            }
            //Case scale input - file scale output
            if(strcmp(argv[1],scaleComment)==0)
            {
                cout<<"Scale_output_file ";
                cin>>scaleFile;
            }
        }

        if(argc<minimumInputInstant)exit_with_help();

        /**Scale*/
        if(strcmp(argv[1],scaleComment)==0)
        {
            ///classifier_v1 svm-scale [options] data_filename
            //Exps:
            //  Scale training file
            //      classifier_v1 svm-scale -u 1 -l -1 -s range train > train_scale.file
            //  Scale testing file
            //      classifier_v1 svm-scale -r range test > test_scale.file
            main_svm_scale(argc,argv);
        }

        /**Train*/
        else if(strcmp(argv[1],trainComment)==0)
        {
            ///classifier_v1 svm-train [options] training_set_file [model_file]
            main_svm_train(argc,argv);
        }

        /**Predict*/
        else if(strcmp(argv[1],predictComment)==0)
        {
            ///classifier_v1 svm-predict [options] test_file model_file output_file
            main_svm_predict(argc,argv);
        }

        /**Cross validation in Multi-case*/
        else if(strcmp(argv[1],crossvalComment)==0)
        {
            ///classifier_v1 svm-crossval [options] valid_file
            main_svm_crossval(argc,argv);
        }
        else
            exit_with_help();
        argc=0;
    }while(inConsole);
    //
    return 0;
}
///
vector<string> splitString(string s, char delim)
{
    vector<string> elems;
    string substring;
    stringstream ss(s);
    while(getline(ss,substring,delim))
    {
        if(!substring.empty())
            elems.push_back(substring);
    }
    return elems;
}
