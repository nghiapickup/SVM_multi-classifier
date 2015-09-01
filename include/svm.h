#ifndef _LIBSVM_H
#define _LIBSVM_H

/**LibSVM 3.20 - Edited 2.1
    @author Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines.
            ACM Transactions on Intelligent Systems and Technology,
            2:27:1--27:27, 2011. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm
    @Edited Nghia NH {nghianh93@gmail.com}
            Adding more methods for multi-class classification.
            I have extended this lib by adding more multi-classifier (represented in my thesis):
            + OAR and its expanding:
                . OAR using Continuous Decision Function by Vapnik, N;
                . OAR fuzzy SVM (FSVM) by Abe, Shigeo.
            + Methods apply in OAO:
                . OAO using Max Wins appling in SVM by KerBel;
                . OAO using FSVM by Abe, Shigeo.
            + DAGSVM by Platt, DDAG by Kijsirikul and Ussivakul, HAH by Lei, Hang Shen.
            + :P TH true-milk (i don't think it work well)
            My code supporting:
            + Probability: OAO, OAR;
            + ONE_CLASS, EPSILON_SVR, NU_SVR: OAO (raw version);
            + Classification (train, predict): All;
            + Cross-validation: All.
*/

#define LIBSVM_VERSION 320
#define EDITED_VERSION 210

#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;

struct svm_node
{
	int index;
	double value;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};

struct svm_tree
{
    int start, end;
    int splitter;
    int split_id;
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */
enum { OAR, OAR_CDF, OAR_FZ, OAO, OAO_FZ, DAG, ADAG, HAH, TH/*so funny with TH-true milk*/}; /* svm_multi_method */

struct svm_parameter
{
    int svm_multi_method;
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

//
// svm_model
//
struct svm_model
{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	double training_time;
	//double building_HAHtree_time; - I am lazy now :"
	struct svm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */

    /* for HAH methods only*/
    int *arrangeHAH;    /* The arrangement of HAH tree*/
    int *splitHAH;      /* HAH spliting order */

	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
};

struct svm_model *svm_group_train(const struct svm_problem *prob, const struct svm_parameter *param);
struct svm_model *svm_pair_train(const struct svm_problem *prob, const struct svm_parameter *param);
struct svm_model *svm_tree_train(const struct svm_problem *prob, const struct svm_parameter *param);
struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_get_svr_probability(const struct svm_model *model);

double svm_OAR_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_OAR_CDF_predict_values(const svm_model *model, const svm_node *x);
double svm_OAR_FZ_predict_values(const svm_model *model, const svm_node *x);
double svm_OAO_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_OAO_FZ_predict_values(const svm_model *model, const svm_node *x);
double svm_DAG_predict_values(const svm_model *model, const svm_node *x);
double svm_ADAG_predict_values(const svm_model *model, const svm_node *x);
double svm_HAH_predict_values(const svm_model *model, const svm_node *x);
double svm_TH_predict_values(const svm_model *model, const svm_node *x);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
