/***********************************************************************/
/*                                                                     */
/*   svm_struct_learn.c                                                */
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

#include "svm_struct_learn.h"
#include "svm_struct_common.h"
#include "../svm_struct_api.h"
#include <assert.h>

#define MAX(x,y)      ((x) < (y) ? (y) : (x))
#define MIN(x,y)      ((x) > (y) ? (y) : (x))

void svm_learn_struct(SAMPLE sample, STRUCT_LEARN_PARM *sparm,
		      LEARN_PARM *lparm, KERNEL_PARM *kparm, 
		      STRUCTMODEL *sm)
{
  int         i,j;
  int         numIt=0;
  long        newconstraints=0, activenum=0; 
  int         opti_round, *opti;
  long        old_numConst=0;
  double      epsilon;
  long        tolerance;
  double      lossval,factor;
  double      margin=0;
  double      slack, *slacks, slacksum;
  long        sizePsi;
  double      *alpha=NULL;
  CONSTSET    cset;
  SVECTOR     *diff=NULL;
  SVECTOR     *fy, *fybar, *f;
  SVECTOR     *slackvec;
  WORD        slackv[2];
  MODEL       *svmModel=NULL;
  KERNEL_CACHE *kcache=NULL;
  LABEL       ybar;
  DOC         *doc;

  long        n=sample.n;
  EXAMPLE     *ex=sample.examples;
  double      rt_total=0.0, rt_opt=0.0;
  long        rt1,rt2;

  init_struct_model(sample,sm,sparm); 
  sizePsi=sm->sizePsi+1;          /* sm must contain size of psi on return */

  /* initialize example selection heuristic */ 
  opti=(int*)my_malloc(n*sizeof(int));
  for(i=0;i<n;i++) {
    opti[i]=0;
  }
  opti_round=0;

  if(sparm->slack_norm == 1) {
    lparm->svm_c=sparm->C;          /* set upper bound C */
    lparm->sharedslack=1;
  }
  else if(sparm->slack_norm == 2) {
    lparm->svm_c=999999999999999.0; /* upper bound C must never be reached */
    lparm->sharedslack=0;
    if(kparm->kernel_type != LINEAR) {
      printf("ERROR: Kernels are not implemented for L2 slack norm!"); 
      fflush(stdout);
      exit(0);
    }
  }
  else {
    printf("ERROR: Slack norm must be L1 or L2!"); fflush(stdout);
    exit(0);
  }


  epsilon=1.0;                    /* start with low precision and
				     increase later */
  tolerance=n/100;                /* increase precision, whenever less
                                     than that number of constraints
                                     is not fulfilled */
  lparm->biased_hyperplane=0;     /* set threshold to zero */

  cset=init_struct_constraints(sample, sm, sparm);
  if(cset.m > 0) {
    alpha=realloc(alpha,sizeof(double)*cset.m);
    for(i=0; i<cset.m; i++) 
      alpha[i]=0;
  }

  /* set initial model and slack variables*/
  svmModel=(MODEL *)my_malloc(sizeof(MODEL));
  svm_learn_optimization(cset.lhs,cset.rhs,cset.m,sizePsi+n,
			 lparm,kparm,NULL,svmModel,alpha);
  add_weight_vector_to_linear_model(svmModel);
  sm->svm_model=svmModel;
  sm->w=svmModel->lin_weights; /* short cut to weight vector */
  sm->dirty = 1;

  printf("Starting Iterations\n");

    /*****************/
   /*** main loop ***/
  /*****************/
  do { /* iteratively increase precision */

    epsilon=MAX(epsilon*0.09999999999,sparm->epsilon);
    if(epsilon == sparm->epsilon)   /* for final precision, find all SV */
      tolerance=0;
    lparm->epsilon_crit=epsilon/2;  /* svm precision must be higher than eps */
    if(struct_verbosity>=1)
      printf("Setting current working precision to %g.\n",epsilon);

    do { /* iteration until (approx) all SV are found for current
            precision and tolerance */
      
      old_numConst=cset.m;
      opti_round++;
      activenum=n;

      do { /* go through examples that keep producing new constraints */

	if(struct_verbosity>=1) { 
	  printf("--Iteration %i (%ld active): ",++numIt,activenum); 
	  fflush(stdout);
	}
	
	for(i=0; i<n; i++) { /*** example loop ***/
	  
	  rt1=get_runtime();
	    
	  if(opti[i] != opti_round) {/* if the example is not shrunk
	                                away, then see if it is necessary to 
					add a new constraint */
	    if(sparm->loss_type == SLACK_RESCALING) 
	      ybar=find_most_violated_constraint_slackrescaling(ex[i].x,
								ex[i].y,sm,
								sparm);
	    else
	      ybar=find_most_violated_constraint_marginrescaling(ex[i].x,
								 ex[i].y,sm,
								 sparm);
	    
	    if(empty_label(ybar)) {
	      if(opti[i] != opti_round) {
		activenum--;
		opti[i]=opti_round; 
	      }
	      if(struct_verbosity>=2)
		printf("no-incorrect-found(%i) ",i);
	      continue;
	    }
	  
	    /**** get psi(y)-psi(ybar) ****/
	    fy=psi(ex[i].x,ex[i].y,sm,sparm);
	    fybar=psi(ex[i].x,ybar,sm,sparm);
	    
	    /**** scale feature vector and margin by loss ****/
	    lossval=loss(ex[i].y,ybar,sparm);
	    if(sparm->slack_norm == 2)
	      lossval=sqrt(lossval);
	    if(sparm->loss_type == SLACK_RESCALING)
	      factor=lossval;
	    else               /* do not rescale vector for */
	      factor=1.0;      /* margin rescaling loss type */
	    for(f=fy;f;f=f->next)
	      f->factor*=factor;
	    for(f=fybar;f;f=f->next)
	      f->factor*=-factor;
	    margin=lossval;

	    /**** create constraint for current ybar ****/
	    append_svector_list(fy,fybar);/* append the two vector lists */
	    doc=create_example(cset.m,0,i+1,1,fy);

	    /**** compute slack for this example ****/
	    slack=0;
	    for(j=0;j<cset.m;j++) 
	      if(cset.lhs[j]->slackid == i+1) {
		if(sparm->slack_norm == 2) /* works only for linear kernel */
		  slack=MAX(slack,cset.rhs[j]
			          -(classify_example(svmModel,cset.lhs[j])
				    -sm->w[sizePsi+i]/(sqrt(2*sparm->C))));
		else
		  slack=MAX(slack,
			   cset.rhs[j]-classify_example(svmModel,cset.lhs[j]));
	      }
	    
	    /**** if `error' add constraint and recompute ****/
	    if((classify_example(svmModel,doc)+slack)<(margin-epsilon)) { 
	      if(struct_verbosity>=2)
		{printf("(%i) ",i); fflush(stdout);}
	      if(struct_verbosity==1)
		{printf("."); fflush(stdout);}
	      
	      /**** resize constraint matrix and add new constraint ****/
	      cset.m++;
	      cset.lhs=realloc(cset.lhs,sizeof(DOC *)*cset.m);
	      if(kparm->kernel_type == LINEAR) {
		diff=add_list_ss(fy); /* store difference vector directly */
		if(sparm->slack_norm == 1) 
		  cset.lhs[cset.m-1]=create_example(cset.m-1,0,i+1,1,
						    copy_svector(diff));
		else if(sparm->slack_norm == 2) {
		  /**** add squared slack variable to feature vector ****/
		  slackv[0].wnum=sizePsi+i;
		  slackv[0].weight=1/(sqrt(2*sparm->C));
		  slackv[1].wnum=0; /*terminator*/
		  slackvec=create_svector(slackv,"",1.0);
		  cset.lhs[cset.m-1]=create_example(cset.m-1,0,i+1,1,
						    add_ss(diff,slackvec));
		  free_svector(slackvec);
		}
		free_svector(diff);
	      }
	      else { /* kernel is used */
		if(sparm->slack_norm == 1) 
		  cset.lhs[cset.m-1]=create_example(cset.m-1,0,i+1,1,
						    copy_svector(fy));
		else if(sparm->slack_norm == 2)
		  exit(1);
	      }
	      cset.rhs=realloc(cset.rhs,sizeof(double)*cset.m);
	      cset.rhs[cset.m-1]=margin;
	      alpha=realloc(alpha,sizeof(double)*cset.m);
	      alpha[cset.m-1]=0;
	      newconstraints++;
	    }
	    else {
	      printf("+"); fflush(stdout); 
	      if(opti[i] != opti_round) {
		activenum--;
		opti[i]=opti_round; 
	      }
	    }

	    free_example(doc,0);
	    free_svector(fy); /* this also free's fybar */
	    free_label(ybar);
	  }

	  /**** get new QP solution ****/
	  if((newconstraints >= sparm->newconstretrain) 
	     || ((newconstraints > 0) && (i == n-1))) {
	    if(struct_verbosity>=1) {
	      printf("*");fflush(stdout);
	    }
	    rt2=get_runtime();
	    free_model(svmModel,0);
	    svmModel=(MODEL *)my_malloc(sizeof(MODEL));
	    /* Always get a new kernel cache. It is not possible to use the
	       same cache for two different training runs */
	    if(kparm->kernel_type != LINEAR)
	      kcache=kernel_cache_init(cset.m,lparm->kernel_cache_size);
	    /* Run the QP solver on cset. */
	    svm_learn_optimization(cset.lhs,cset.rhs,cset.m,sizePsi+n,
				   lparm,kparm,kcache,svmModel,alpha);
	    if(kcache)
	      kernel_cache_cleanup(kcache);
	    /* Always add weight vector, in case part of the kernel is
	       linear. If not, ignore the weight vector since its
	       content is bogus. */
	    add_weight_vector_to_linear_model(svmModel);
	    sm->svm_model=svmModel;
	    sm->w=svmModel->lin_weights; /* short cut to weight vector */
	    sm->dirty = 1;
	    rt_opt+=MAX(get_runtime()-rt2,0);
	    
	    newconstraints=0;
	  }	

	  rt_total+=MAX(get_runtime()-rt1,0);
	} /* end of example loop */

	if(struct_verbosity>=1)
	  printf("(NumConst=%d, SV=%ld, Eps=%.4f)\n",cset.m,svmModel->sv_num-1,
		 svmModel->maxdiff);

      } while(activenum > 0);   /* repeat until all examples produced no
				   constraint at least once */

    } while((cset.m - old_numConst) > tolerance) ;

  } while(epsilon > sparm->epsilon);  

  if(struct_verbosity>=1) {
    /**** compute sum of slacks ****/
    slacks=(double *)my_malloc(sizeof(double)*(n+1));
    for(i=0; i<=n; i++) { 
      slacks[i]=0;
    }
    if(sparm->slack_norm == 1) {
      for(j=0;j<cset.m;j++) 
	slacks[cset.lhs[j]->slackid]=MAX(slacks[cset.lhs[j]->slackid],
			   cset.rhs[j]-classify_example(svmModel,cset.lhs[j]));
      }
    else if(sparm->slack_norm == 2) {
      for(j=0;j<cset.m;j++) 
	slacks[cset.lhs[j]->slackid]=MAX(slacks[cset.lhs[j]->slackid],
		cset.rhs[j]
	         -(classify_example(svmModel,cset.lhs[j])
		   -sm->w[sizePsi+cset.lhs[j]->slackid-1]/(sqrt(2*sparm->C))));
    }
    slacksum=0;
    for(i=0; i<=n; i++)  
      slacksum+=slacks[i];
    free(slacks);

    printf("Final epsilon on KKT-Conditions: %.5f\n",
	   MAX(svmModel->maxdiff,epsilon));
    printf("Total number of constraints added: %i\n",(int)cset.m);
    if(sparm->slack_norm == 1) {
      printf("Number of SV: %ld \n",svmModel->sv_num-1);
      printf("Number of non-zero slack variables: %ld (out of %ld)\n",
	     svmModel->at_upper_bound,n);
      printf("Norm of weight vector: |w|=%.5f\n",
	     model_length_s(svmModel,kparm));
    }
    else if(sparm->slack_norm == 2){ 
      printf("Number of SV: %ld (including %ld at upper bound)\n",
	     svmModel->sv_num-1,svmModel->at_upper_bound);
      printf("Norm of weight vector (including L2-loss): |w|=%.5f\n",
	     model_length_s(svmModel,kparm));
    }
    printf("Sum of slack variables: sum(xi_i)=%.5f\n",slacksum);
    printf("Norm of longest difference vector: ||Psi(x,y)-Psi(x,ybar)||=%.5f\n",
	   length_of_longest_document_vector(cset.lhs,cset.m,kparm));
    printf("Runtime in cpu-seconds: %.2f (%.2f%% for SVM optimization)\n",
	   rt_total/100.0, 100.0*rt_opt/rt_total);
  }
  if(struct_verbosity>=4)
    printW(sm->w,sizePsi,n,lparm->svm_c);

  if(svmModel) {
    sm->svm_model=copy_model(svmModel);
    sm->w=sm->svm_model->lin_weights; /* short cut to weight vector */
    sm->dirty = 1;
  }

  print_struct_learning_stats(sample,sm,cset,alpha,sparm);

  if(svmModel)
    free_model(svmModel,0);
  free(alpha); 
  free(opti); 
  free(cset.rhs); 
  for(i=0;i<cset.m;i++) 
    free_example(cset.lhs[i],1);
  free(cset.lhs);
}



