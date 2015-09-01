#ifndef SVM_SCALE_H
#define SVM_SCALE_H

#include <iostream>
#include <fstream>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <string>
using namespace std;

//Define Gobal variables
extern string scaleFile;

//Define Methods
extern void exit_with_help_scale();
extern int main_svm_scale(int argc,char **argv);

#endif // SVM_SCALE_H
