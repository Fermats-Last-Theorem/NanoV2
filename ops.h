#pragma once
#include "arena.h"

ll add(ll a, ll b) {
    ll id = get_node(nodes[a].r, nodes[a].c, 1, a, b);
    ll sz = nodes[a].r * nodes[a].c;
    for(ll i=0; i<sz; ++i) nodes[id].val[i] = nodes[a].val[i] + nodes[b].val[i];
    return id;
}

ll mul(ll a, ll b) {
    ll id = get_node(nodes[a].r, nodes[a].c, 2, a, b);
    ll sz = nodes[a].r * nodes[a].c;
    for(ll i=0; i<sz; ++i) nodes[id].val[i] = nodes[a].val[i] * nodes[b].val[i];
    return id;
}

void mm_w(ll id, ll a, ll b, ll s, ll e) {
    ll k_sz = nodes[a].c;
    ll c_sz = nodes[b].c;
    for(ll i=s; i<e; ++i) {
        for(ll j=0; j<c_sz; ++j) {
            float sum = 0;
            for(ll k=0; k<k_sz; ++k) sum += nodes[a].val[i*k_sz + k] * nodes[b].val[k*c_sz + j];
            nodes[id].val[i*c_sz + j] = sum;
        }
    }
}

ll matmul(ll a, ll b) {
    ll id = get_node(nodes[a].r, nodes[b].c, 3, a, b);
    ll t_cnt = 4;
    thread t[4];
    ll block = nodes[a].r / t_cnt;
    for(ll i=0; i<t_cnt; ++i) {
        ll s = i * block;
        ll e = (i == t_cnt - 1) ? nodes[a].r : s + block;
        t[i] = thread(mm_w, id, a, b, s, e);
    }
    for(ll i=0; i<t_cnt; ++i) t[i].join();
    return id;
}

ll relu(ll a) {
    ll id = get_node(nodes[a].r, nodes[a].c, 4, a, -1);
    ll sz = nodes[a].r * nodes[a].c;
    for(ll i=0; i<sz; ++i) nodes[id].val[i] = nodes[a].val[i] > 0 ? nodes[a].val[i] : 0;
    return id;
}