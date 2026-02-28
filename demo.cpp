#include <iostream>
#include <thread>
#include <cmath>
#include <cstring>
#include <random>

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
    ll k_sz = nodes[a].c, c_sz = nodes[b].c;
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

ll sigmoid(ll a) {
    ll id = get_node(nodes[a].r, nodes[a].c, 5, a, -1);
    ll sz = nodes[a].r * nodes[a].c;
    for(ll i=0; i<sz; ++i) nodes[id].val[i] = 1.0f / (1.0f + exp(-nodes[a].val[i]));
    return id;
}

ll bce(ll p, ll y) {
    ll id = get_node(1, 1, 6, p, y);
    ll sz = nodes[p].r * nodes[p].c;
    float loss = 0;
    for(ll i=0; i<sz; ++i) {
        float val = nodes[p].val[i];
        if(val < 1e-9) val = 1e-9;
        if(val > 1.0f - 1e-9) val = 1.0f - 1e-9;
        loss -= (nodes[y].val[i] * log(val) + (1.0f - nodes[y].val[i]) * log(1.0f - val));
    }
    nodes[id].val[0] = loss / sz;
    return id;
}

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
        } else if(nodes[u].op == 5) {
            for(ll j=0; j<n; ++j) {
                float sig = nodes[u].val[j];
                nodes[p1].grad[j] += sig * (1.0f - sig) * nodes[u].grad[j];
            }
        } else if(nodes[u].op == 6) {
            ll p_sz = nodes[p1].r * nodes[p1].c;
            for(ll j=0; j<p_sz; ++j) {
                float p = nodes[p1].val[j];
                float y = nodes[p2].val[j];
                if(p < 1e-9) p = 1e-9;
                if(p > 1.0f - 1e-9) p = 1.0f - 1e-9;
                nodes[p1].grad[j] += (nodes[u].grad[0] * (p - y)) / (p * (1.0f - p) * p_sz);
            }
        }
    }
}

float X_d[2][1000];
float Y_d[1000];

void gen_data() {
    for(ll i=0; i<1000; ++i) {
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        X_d[0][i] = x1;
        X_d[1][i] = x2;
        Y_d[i] = (x1 * x2 > 0) ? 1.0f : 0.0f;
    }
}

int main() {
    srand(42);
    gen_data();

    // Persistent network weights (allocated outside the training loop)
    ll w1 = get_node(2, 4, 0, -1, -1);
    for(ll i=0; i<8; ++i) nodes[w1].val[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    
    ll w2 = get_node(4, 1, 0, -1, -1);
    for(ll i=0; i<4; ++i) nodes[w2].val[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;

    ll n_lim = n_ptr; // Snapshot the pointer states
    ll m_lim = m_ptr;
    ll b_sz = 32;

    for(ll ep=0; ep<=50; ++ep) {
        float t_loss = 0;
        float t_acc = 0;
        ll b_cnt = 0;

        for(ll i=0; i<1000; i+=b_sz) {
            ll cur_b = min(b_sz, 1000LL - i);
            
            ll x = get_node(cur_b, 2, 0, -1, -1);
            ll y = get_node(cur_b, 1, 0, -1, -1);

            for(ll j=0; j<cur_b; ++j) {
                nodes[x].val[j*2] = X_d[0][i+j];
                nodes[x].val[j*2 + 1] = X_d[1][i+j];
                nodes[y].val[j] = Y_d[i+j];
            }

            // Forward Pass
            ll h1 = matmul(x, w1);
            ll a1 = relu(h1);
            ll out = matmul(a1, w2);
            ll p = sigmoid(out);
            ll loss = bce(p, y);

            // Backward Pass
            backward(loss);

            // SGD Step (only update persistent weights below n_lim)
            for(ll k=0; k<n_lim; ++k) {
                if(nodes[k].op == 0) {
                    ll sz = nodes[k].r * nodes[k].c;
                    for(ll idx=0; idx<sz; ++idx) nodes[k].val[idx] -= 0.1f * nodes[k].grad[idx];
                }
            }

            // Zero Gradients
            for(ll k=0; k<n_lim; ++k) {
                ll sz = nodes[k].r * nodes[k].c;
                memset(nodes[k].grad, 0, sizeof(float) * sz);
            }

            // Metrics
            t_loss += nodes[loss].val[0];
            for(ll j=0; j<cur_b; ++j) {
                float p_val = nodes[p].val[j];
                float target = nodes[y].val[j];
                if((p_val > 0.5f && target == 1.0f) || (p_val <= 0.5f && target == 0.0f)) {
                    t_acc += 1.0f;
                }
            }
            b_cnt++;

            // O(1) Instant Memory Deallocation. The graph is wiped, the weights remain.
            n_ptr = n_lim;
            m_ptr = m_lim;
        }

        if(ep % 5 == 0) {
            cout << "Epoch " << ep << ", Loss: " << (t_loss / b_cnt) 
                 << ", Acc: " << (t_acc / 1000.0f) << "\n";
        }
    }

    return 0;
}