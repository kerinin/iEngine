/***********************************************************************/
/*                                                                     */
/*   svm_struct_learn.h                                                */
/*                                                                     */
/*   Basic algorithm for learning structured outputs (e.g. parses,     */
/*   sequences, multi-label classification) with a Support Vector      */ 
/*   Machine.                                                          */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#ifndef SVM_STRUCT_LEARN
#define SVM_STRUCT_LEARN

#include "../svm_light/svm_common.h"
#include "../svm_light/svm_learn.h"
#include "svm_struct_common.h" 
#include "../svm_struct_api_types.h" 

#define  SLACK_RESCALING    1
#define  MARGIN_RESCALING   2

void svm_learn_struct(SAMPLE sample, STRUCT_LEARN_PARM *sparm,
		      LEARN_PARM *lparm, KERNEL_PARM *kparm, 
		      STRUCTMODEL *sm);

#endif


