#pragma once
#include <iostream>
#include <thread>
#include <cmath>
#include <cstring>

using namespace std;
typedef long long ll;

const ll MAX_MEM = 50000000;
const ll MAX_NODES = 5000000;

float mem[MAX_MEM];
ll m_ptr = 0;

float* get_mem(ll sz) {
    float* p = &mem[m_ptr];
    m_ptr += sz;
    return p;
}

struct Node {
    ll r, c;
    float* val;
    float* grad;
    ll p1, p2;
    int op; 
};

Node nodes[MAX_NODES];
ll n_ptr = 0;

ll get_node(ll r, ll c, int op, ll p1, ll p2) {
    ll id = n_ptr++;
    nodes[id].r = r;
    nodes[id].c = c;
    nodes[id].val = get_mem(r * c);
    nodes[id].grad = get_mem(r * c);
    memset(nodes[id].val, 0, sizeof(float) * r * c);
    memset(nodes[id].grad, 0, sizeof(float) * r * c);
    nodes[id].op = op;
    nodes[id].p1 = p1;
    nodes[id].p2 = p2;
    return id;
}