#include "autograd.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// A node in the graph for topological sort
typedef struct TopoNode {
    Tensor *tensor;
    bool visited;
} TopoNode;

// A list of nodes
typedef struct NodeList {
    TopoNode **nodes;
    int count;
    int capacity;
} NodeList;

void nodelist_add(NodeList *list, TopoNode *node) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->nodes = realloc(list->nodes, list->capacity * sizeof(TopoNode *));
    }
    list->nodes[list->count++] = node;
}

void find_all_tensors(NodeList *all_nodes, Tensor *current) {
    for (int i = 0; i < all_nodes->count; i++) {
        if (all_nodes->nodes[i]->tensor == current)
            return;
    }
    TopoNode *new_node = malloc(sizeof(TopoNode));
    new_node->tensor = current;
    new_node->visited = false;
    nodelist_add(all_nodes, new_node);

    if (current->grad_fn == cross_entropy_backward) {
        find_all_tensors(all_nodes, (Tensor *)current->ctx[0]);
    } else if (current->ctx) {
        for (int i = 0; i < current->ctx_size; i++) {
            find_all_tensors(all_nodes, (Tensor *)current->ctx[i]);
        }
    }
}

void build_topo_sort(NodeList *sorted, NodeList *all_nodes, Tensor *t) {
    TopoNode *node = NULL;
    for (int i = 0; i < all_nodes->count; i++) {
        if (all_nodes->nodes[i]->tensor == t) {
            node = all_nodes->nodes[i];
            break;
        }
    }
    if (node == NULL || node->visited) {
        return;
    }
    node->visited = true;

    if (t->grad_fn == cross_entropy_backward) {
        build_topo_sort(sorted, all_nodes, (Tensor *)t->ctx[0]);
    } else if (t->ctx) {
        for (int i = 0; i < t->ctx_size; i++) {
            build_topo_sort(sorted, all_nodes, (Tensor *)t->ctx[i]);
        }
    }
    nodelist_add(sorted, node);
}

void tensor_backward(Tensor *t) {
    if (!t->requires_grad) {
        printf("Cannot call backward on a tensor that does not require grad\n");
        return;
    }

    NodeList *all_nodes = malloc(sizeof(NodeList));
    all_nodes->capacity = 16;
    all_nodes->count = 0;
    all_nodes->nodes = malloc(all_nodes->capacity * sizeof(TopoNode *));
    find_all_tensors(all_nodes, t);

    NodeList *sorted = malloc(sizeof(NodeList));
    sorted->capacity = all_nodes->count;
    sorted->count = 0;
    sorted->nodes = malloc(sorted->capacity * sizeof(TopoNode *));
    build_topo_sort(sorted, all_nodes, t);

    if (t->grad == NULL) {
        size_t size = tensor_size(t);
        float *grad_data = (float *)malloc(size * sizeof(float));
        for (size_t i = 0; i < size; i++) {
            grad_data[i] = 1.0f;
        }
        t->grad = tensor_create(grad_data, t->shape, t->ndim, false);
        free(grad_data);
    }

    for (int i = sorted->count - 1; i >= 0; i--) {
        Tensor *current = sorted->nodes[i]->tensor;
        if (current->grad_fn) {
            current->grad_fn(current);
        }
    }

    for (int i = 0; i < all_nodes->count; i++) {
        free(all_nodes->nodes[i]);
    }
    free(all_nodes->nodes);
    free(all_nodes);
    free(sorted->nodes);
    free(sorted);
}