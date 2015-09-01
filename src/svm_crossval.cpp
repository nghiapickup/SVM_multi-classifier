/**
    @Author Nghia NH {nghianh93@gmail.com}
*/
#include "svm_crossval.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
///Define cross validation scale
///recommended by Hsu, Chih-Wei and Lin, Chih-Jen
///C from 2^-2 to 2^12
///gamma from 2^-10 to 2^4
#define min_c_param 0.25    //2^-3
#define max_c_param 4096    //2^12
#define progression_c_param 2
//
#define min_gamma_param 0.0009765625    //2^-10
#define max_gamma_param 16    //2^4
#define progression_gamma_param 2

void print_null_crossval(const char *s) {}

void exit_with_help_crossval()
{
	printf("Using cross-validation to find the best params\n"
	"Usage:  svm-crossval [options] valid_file output_file\n"
	"Const:\n"
	"svm_type : C-SVC\n"
	"Choosing ranges of param: Auto\n"
	"options:\n"
    "-l svm_type : set multi-classifier method of SVM (default 0)\n"
	"	0 -- OAR		(One-Against-Rest)\n"
	"	1 -- OAR_CDF	(OAR using Continuous Decision Function by Vapnik, N)\n"
	"	2 -- OAR_FZ		(OAR fuzzy SVM (FSVM) by Abe, Shigeo)\n"
	"	3 -- OAO		(One-Against-One using Max Wins appling in SVM by KerBel)\n"
	"	4 -- OAO_FZ		(OAO using FSVM by Abe, Shigeo)\n"
	"	5 -- DAG		(DAGSVM by Platt)\n"
	"	6 -- ADAG		(DDAG by Kijsirikul and Ussivakul)\n"
	"	7 -- HAH		(HAH by Lei, Hang Shen)\n"
	"	8 -- TH		    (TH true-milk - LOL)\n"//Just for fun, basic train & test, don't use it in others option.
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
    "-v n: n-fold cross validation mode (default 10)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error_crossval(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_crossval_command_line(int argc, char **argv, char *input_file_name, char *output_file_name);
void read_crossval_problem(const char *filename);
double do_multi_cross_validation();

struct svm_parameter param_crossval;		// set by parse_command_line
struct svm_problem prob_crossval;		// set by read_problem
struct svm_model *model_crossval;
struct svm_node *x_space_crossval;
int nr_fold_crossval;

static char *line = NULL;
static int max_line_len;

static char* readline_crossval(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

static const char *svm_multi_method_table[] =
{
	"OAR", "OAR_CDF", "OAR_FZ", "OAO", "OAO_FZ", "DAG", "ADAG", "HAH", "TH",NULL
};

int main_svm_crossval(int argc, char **argv)
{
	char input_file_name[1024];
	char output_file_name[1024];
	FILE *output;
	const char *error_msg;

	parse_crossval_command_line(argc, argv, input_file_name, output_file_name);
	read_crossval_problem(input_file_name);
	//
	//Check papameter
	error_msg = svm_check_parameter(&prob_crossval,&param_crossval);
    printf("Cross Validation\n");
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

    output = fopen(output_file_name,"w");
	///cross_validation
	{
	    fprintf(output,"svm_multi_method %s\n",svm_multi_method_table[param_crossval.svm_multi_method]);
	    fprintf(output,"nr_fold %d\n",nr_fold_crossval);
	    double c_target = min_c_param, gamma_target = min_gamma_param;
	    double temp_target = 0, temp = 0;

	    //kernel_type==LINEAR
	    if(param_crossval.kernel_type==LINEAR)
        {
            printf("LINEAR kernel\nScaling C parameter\n\n");
            fprintf(output,"kernel LINEAR\n\n");
            for(double i = min_c_param; i<= max_c_param; i*=progression_c_param)
            {
                param_crossval.C = i;
                temp = do_multi_cross_validation();
                fprintf(output,"%g  %g\n",i,temp);
                if(temp > temp_target)
                {
                    temp_target = temp;
                    c_target = i;
                }
            }
            fprintf(output,"Best C parameter: [%g]\n",c_target);
            printf("Best C parameter: [%g]\n",c_target);
        }

        //kernel_type==RBF
        else if (param_crossval.kernel_type==RBF)
        {
            printf("RBF kernel\nScaling C and gamma parameters\n\n");
            fprintf(output,"kernel RBF\n\n");
            for(double i = min_c_param; i<= max_c_param; i*=progression_c_param)
                for(double j=min_gamma_param; j<= max_gamma_param; j*=progression_gamma_param)
                {
                    param_crossval.C = i;
                    param_crossval.gamma = j;
                    temp = do_multi_cross_validation();
                    fprintf(output,"%g  %g  %g\n",i,j,temp);
                    if(temp > temp_target)
                    {
                        temp_target = temp;
                        c_target = i;
                        gamma_target = j;
                    }
                }
            fprintf(output,"Best C and Gamma parameters: [%g  %g]\n",c_target,gamma_target);
            printf("Best C and Gamma parameters: [%g  %g]\n",c_target,gamma_target);
        }
        else
        {
            printf("Do not supporting this kernel_type now !");
        }
	}
	svm_destroy_param(&param_crossval);
	free(prob_crossval.y);
	free(prob_crossval.x);
	free(x_space_crossval);
	free(line);
	fclose(output);
	return 0;
}

double do_multi_cross_validation()
{
	int i;
	int total_correct = 0;
	double result=0;
	double *target = Malloc(double,prob_crossval.l);
	svm_cross_validation(&prob_crossval,&param_crossval,nr_fold_crossval,target);
	///param_crossval.svm_type == C-SVC
	{
		for(i=0;i<prob_crossval.l;i++)
			if(target[i] == prob_crossval.y[i])
				++total_correct;
        result=100.0*total_correct/prob_crossval.l;
		printf("Cross Validation Accuracy = %g%%\n",result);
	}
	free(target);
	return result;
}

void parse_crossval_command_line(int argc, char **argv, char *input_file_name, char *output_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param_crossval.svm_multi_method = OAO;
	param_crossval.svm_type = C_SVC;
	param_crossval.kernel_type = RBF;
	param_crossval.degree = 3;
	param_crossval.gamma = 0;	// 1/num_features
	param_crossval.coef0 = 0;
    param_crossval.C = 1;
	param_crossval.cache_size = 100;
	param_crossval.nr_weight = 0;
	param_crossval.weight_label = NULL;
	param_crossval.weight = NULL;
    nr_fold_crossval = 10;
	// default values _ do not use
	param_crossval.nu = 0.5;
	param_crossval.eps = 1e-3;
	param_crossval.p = 0.1;
	param_crossval.shrinking = 1;
	param_crossval.probability = 0;


	// parse options
	for(i=2;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help_crossval();
		switch(argv[i-1][1])
		{
            case 'l':
				param_crossval.svm_multi_method = atoi(argv[i]);
				break;
			case 't':
				param_crossval.kernel_type = atoi(argv[i]);
				break;
			case 'm':
				param_crossval.cache_size = atof(argv[i]);
				break;
			case 'q':
				print_func = &print_null_crossval;
				i--;
				break;
			case 'v':
				nr_fold_crossval = atoi(argv[i]);
				if(nr_fold_crossval < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help_crossval();
				}
				break;
			case 'w':
				++param_crossval.nr_weight;
				param_crossval.weight_label = (int *)realloc(param_crossval.weight_label,sizeof(int)*param_crossval.nr_weight);
				param_crossval.weight = (double *)realloc(param_crossval.weight,sizeof(double)*param_crossval.nr_weight);
				param_crossval.weight_label[param_crossval.nr_weight-1] = atoi(&argv[i-1][2]);
				param_crossval.weight[param_crossval.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help_crossval();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc)
		exit_with_help_crossval();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(output_file_name,argv[i+1]);
    else
        exit_with_help_crossval();
}

// read in a problem (in svmlight format)

void read_crossval_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob_crossval.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline_crossval(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob_crossval.l;
	}
	rewind(fp);

	prob_crossval.y = Malloc(double,prob_crossval.l);
	prob_crossval.x = Malloc(struct svm_node *,prob_crossval.l);
	x_space_crossval = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob_crossval.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline_crossval(fp);
		prob_crossval.x[i] = &x_space_crossval[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error_crossval(i+1);

		prob_crossval.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error_crossval(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space_crossval[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space_crossval[j].index <= inst_max_index)
				exit_input_error_crossval(i+1);
			else
				inst_max_index = x_space_crossval[j].index;

			errno = 0;
			x_space_crossval[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error_crossval(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space_crossval[j++].index = -1;
	}

	if(param_crossval.gamma == 0 && max_index > 0)
		param_crossval.gamma = 1.0/max_index;

	if(param_crossval.kernel_type == PRECOMPUTED)
		for(i=0;i<prob_crossval.l;i++)
		{
			if (prob_crossval.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob_crossval.x[i][0].value <= 0 || (int)prob_crossval.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}
