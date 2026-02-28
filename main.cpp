#include "autograd.h"

void step(float lr, ll max_w) {
    for(ll i=0; i<max_w; ++i) {
        if(nodes[i].op == 0) {
            ll n = nodes[i].r * nodes[i].c;
            for(ll j=0; j<n; ++j) nodes[i].val[j] -= lr * nodes[i].grad[j];
        }
    }
}

void zero(ll max_w) {
    for(ll i=0; i<max_w; ++i) {
        ll n = nodes[i].r * nodes[i].c;
        memset(nodes[i].grad, 0, sizeof(float) * n);
    }
}

int main() {
    ll x = get_node(32, 2, 0, -1, -1);
    ll w1 = get_node(2, 4, 0, -1, -1);
    ll w2 = get_node(4, 1, 0, -1, -1);
    
    ll limit = n_ptr;
    
    for(ll epoch=0; epoch<100; ++epoch) {
        ll h1 = matmul(x, w1);
        ll a1 = relu(h1);
        ll out = matmul(a1, w2);
        
        backward(out);
        step(0.01f, limit);
        zero(limit);
        
        m_ptr = nodes[limit].val - mem; 
        n_ptr = limit;
    }
    
    return 0;
}