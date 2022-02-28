#ifndef __MEASURE_LIB_H__
#define __MEASURE_LIB_H__

typedef struct
{
    unsigned short emiss;
    unsigned short relHum;
    unsigned short distance;
    short reflectedTemper;
    short atmosphericTemper;
    unsigned short modifyK;
    short modifyB;
}guide_measure_external_param_t;


int guide_measure_convertgray2temper(short *pGray,  unsigned char *pParamLine, int len, guide_measure_external_param_t *pParamExt, float *pTemper);
int guide_measure_converttemper2gray(float *pTemper, char *pParamLine, int len, guide_measure_external_param_t *pParamExt, short *pGray);

#endif

