/*
 * @Author: RyanCLQ
 * @Date: 2023-06-06 16:03:15
 * @LastEditors: RyanCLQ
 * @LastEditTime: 2023-06-13 11:13:41
 * @Description: 请填写简介
 */
#ifndef SPEED_C
#define SPEED_C

#include <sys/times.h>
#include <unistd.h>
#include <signal.h>
#include <stdio.h>

#define TM_START        0
#define TM_STOP         1

#define START        0
#define STOP         1

extern volatile int run;

void alarmed(int sig);
double app_tminterval(int stop);
double Time_F(int s);
size_t test_func_loop(size_t flag);
//performance test for enc 
void performance_test_enc(size_t (*func_loop)(size_t), size_t size_arr[], size_t num, int sec);

void performance_test(size_t (*func_loop)(size_t), size_t size_arr[], size_t num);
void performance_test_mode(size_t (*func_loop)(), int sec);
//performance test for sign and verify
void performance_test_sv(size_t (*func_loop_sign)(size_t), size_t (*func_loop_verify)(size_t), size_t size_arr[], size_t num, int sec);

#endif
