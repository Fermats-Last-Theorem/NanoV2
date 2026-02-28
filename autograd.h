#pragma once
#include "ops.h"

ll topo[MAX_NODES];
ll t_sz = 0;
bool vis[MAX_NODES];
ll st[MAX_NODES];

void dfs(ll root) {
    t_sz = 0;
    memset(vis, 0, sizeof(vis));
    ll s_ptr = 0;
    st[s_ptr++] = root;
    
    while(s_ptr > 0) {
        ll u = st[s_ptr - 1];
        if(vis[u]) {
            s_ptr--;
            continue;
        }
        bool push = false;
        ll p1 = nodes[u].p1, p2 = nodes[u].p2;
        if(p1 != -1 && !vis[p1]) { st[s_ptr++] = p1; push = true; }
        if(p2 != -1 && !vis[p2]) { st[s_ptr++] = p2; push = true; }
        if(!push) {
            vis[u] = true;
            topo[t_sz++] = u;
            s_ptr--;
        }
    }
}

void b_mm_a(ll a, ll b, ll out, ll s, ll e) {
    ll k_sz = nodes[a].c, c_sz = nodes[b].c;
    for(ll i=s; i<e; ++i) {
        for(ll k=0; k<k_sz; ++k) {
            float sum = 0;
            for(ll j=0; j<c_sz; ++j) sum += nodes[out].grad[i*c_sz + j] * nodes[b].val[k*c_sz + j];
            nodes[a].grad[i*k_sz + k] += sum;
        }
    }
}

void b_mm_b(ll a, ll b, ll out, ll s, ll e) {
    ll r_sz = nodes[a].r, c_sz = nodes[b].c;
    for(ll k=s; k<e; ++k) {
        for(ll j=0; j<c_sz; ++j) {
            float sum = 0;
            for(ll i=0; i<r_sz; ++i) sum += nodes[a].val[i*nodes[a].c + k] * nodes[out].grad[i*c_sz + j];
            nodes[b].grad[k*c_sz + j] += sum;
        }
    }
}

void backward(ll root) {
    dfs(root);
    ll sz = nodes[root].r * nodes[root].c;
    for(ll i=0; i<sz; ++i) nodes[root].grad[i] = 1.0f;
    
    for(ll i = t_sz - 1; i >= 0; --i) {
        ll u = topo[i];
        ll p1 = nodes[u].p1, p2 = nodes[u].p2;
        ll n = nodes[u].r * nodes[u].c;
        
        if(nodes[u].op == 1) {
            for(ll j=0; j<n; ++j) {
                nodes[p1].grad[j] += nodes[u].grad[j];
                nodes[p2].grad[j] += nodes[u].grad[j];
            }
        } else if(nodes[u].op == 2) {
            for(ll j=0; j<n; ++j) {
                nodes[p1].grad[j] += nodes[u].grad[j] * nodes[p2].val[j];
                nodes[p2].grad[j] += nodes[u].grad[j] * nodes[p1].val[j];
            }
        } else if(nodes[u].op == 3) {
            ll t_cnt = 4;
            thread t1[4];
            ll b1 = nodes[p1].r / t_cnt;
            for(ll w=0; w<t_cnt; ++w) {
                ll s = w * b1;
                ll e = (w == t_cnt - 1) ? nodes[p1].r : s + b1;
                t1[w] = thread(b_mm_a, p1, p2, u, s, e);
            }
            for(ll w=0; w<t_cnt; ++w) t1[w].join();

            thread t2[4];
            ll b2 = nodes[p1].c / t_cnt;
            for(ll w=0; w<t_cnt; ++w) {
                ll s = w * b2;
                ll e = (w == t_cnt - 1) ? nodes[p1].c : s + b2;
                t2[w] = thread(b_mm_b, p1, p2, u, s, e);
            }
            for(ll w=0; w<t_cnt; ++w) t2[w].join();
        } else if(nodes[u].op == 4) {
            for(ll j=0; j<n; ++j) {
                nodes[p1].grad[j] += (nodes[p1].val[j] > 0) ? nodes[u].grad[j] : 0;
            }
        }
    }
}