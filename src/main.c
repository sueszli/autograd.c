#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONCAT(a, b) a ## b
#define CONCAT_EXPAND(a, b) CONCAT(a, b)
#define UNIQUE_NAME(base) CONCAT_EXPAND(base, __LINE__)

#define defer(block) \
    void UNIQUE_NAME(__cleanup_)(int *ptr __attribute__((unused))) { block } \
    int UNIQUE_NAME(__defer_var_) __attribute__((cleanup(UNIQUE_NAME(__cleanup_)))) = 0

typedef struct {
    char *name;
    int *data;
    size_t data_size;
    struct {
        double *values;
        size_t count;
    } metrics;
} ComplexStruct;

typedef struct Node {
    int value;
    struct Node *next;
    char *label;
} Node;

typedef struct {
    Node **buckets;
    size_t bucket_count;
    size_t total_items;
    char *table_name;
} HashTable;

ComplexStruct* create_complex_struct(const char *name, size_t data_size, size_t metrics_count) {
    ComplexStruct *cs = malloc(sizeof(ComplexStruct));
    if (!cs) return NULL;
    
    cs->name = malloc(strlen(name) + 1);
    if (!cs->name) {
        free(cs);
        return NULL;
    }
    strcpy(cs->name, name);
    
    cs->data = malloc(data_size * sizeof(int));
    if (!cs->data) {
        free(cs->name);
        free(cs);
        return NULL;
    }
    cs->data_size = data_size;
    
    cs->metrics.values = malloc(metrics_count * sizeof(double));
    if (!cs->metrics.values) {
        free(cs->data);
        free(cs->name);
        free(cs);
        return NULL;
    }
    cs->metrics.count = metrics_count;
    
    for (size_t i = 0; i < data_size; i++) {
        cs->data[i] = (int)(i * 42);
    }
    
    for (size_t i = 0; i < metrics_count; i++) {
        cs->metrics.values[i] = i * 3.14159;
    }
    
    return cs;
}

void free_complex_struct(ComplexStruct *cs) {
    if (!cs) return;
    printf("Freeing ComplexStruct: %s\n", cs->name ? cs->name : "unnamed");
    free(cs->metrics.values);
    free(cs->data);
    free(cs->name);
    free(cs);
}

Node* create_node(int value, const char *label) {
    Node *node = malloc(sizeof(Node));
    if (!node) return NULL;
    
    node->value = value;
    node->next = NULL;
    node->label = malloc(strlen(label) + 1);
    if (!node->label) {
        free(node);
        return NULL;
    }
    strcpy(node->label, label);
    
    return node;
}

void free_node_chain(Node *head) {
    while (head) {
        Node *next = head->next;
        printf("Freeing node: %s (value: %d)\n", head->label, head->value);
        free(head->label);
        free(head);
        head = next;
    }
}

HashTable* create_hash_table(size_t bucket_count, const char *name) {
    HashTable *ht = malloc(sizeof(HashTable));
    if (!ht) return NULL;
    
    ht->buckets = calloc(bucket_count, sizeof(Node*));
    if (!ht->buckets) {
        free(ht);
        return NULL;
    }
    
    ht->table_name = malloc(strlen(name) + 1);
    if (!ht->table_name) {
        free(ht->buckets);
        free(ht);
        return NULL;
    }
    strcpy(ht->table_name, name);
    
    ht->bucket_count = bucket_count;
    ht->total_items = 0;
    
    return ht;
}

void free_hash_table(HashTable *ht) {
    if (!ht) return;
    printf("Freeing HashTable: %s\n", ht->table_name);
    
    for (size_t i = 0; i < ht->bucket_count; i++) {
        if (ht->buckets[i]) {
            free_node_chain(ht->buckets[i]);
        }
    }
    
    free(ht->buckets);
    free(ht->table_name);
    free(ht);
}

void demonstrate_complex_allocations() {
    printf("\n=== Complex Struct Allocation Demo ===\n");
    
    ComplexStruct *struct1 = create_complex_struct("DataProcessor", 10, 5);
    defer(free_complex_struct(struct1););
    
    ComplexStruct *struct2 = create_complex_struct("MetricsCollector", 20, 8);
    defer(free_complex_struct(struct2););
    
    if (struct1 && struct2) {
        printf("Created structs: %s and %s\n", struct1->name, struct2->name);
        printf("Struct1 data[0]: %d, metrics[0]: %.2f\n", 
               struct1->data[0], struct1->metrics.values[0]);
        printf("Struct2 data[5]: %d, metrics[3]: %.2f\n", 
               struct2->data[5], struct2->metrics.values[3]);
    }
}

void demonstrate_linked_list() {
    printf("\n=== Linked List Demo ===\n");
    
    Node *head = create_node(100, "head_node");
    defer(free_node_chain(head););
    
    if (head) {
        head->next = create_node(200, "second_node");
        if (head->next) {
            head->next->next = create_node(300, "third_node");
            if (head->next->next) {
                head->next->next->next = create_node(400, "tail_node");
            }
        }
        
        printf("Created linked list starting with: %s\n", head->label);
        Node *current = head;
        while (current) {
            printf("  -> %s: %d\n", current->label, current->value);
            current = current->next;
        }
    }
}

void demonstrate_hash_table() {
    printf("\n=== Hash Table Demo ===\n");
    
    HashTable *table = create_hash_table(8, "UserHashTable");
    defer(free_hash_table(table););
    
    if (table) {
        table->buckets[0] = create_node(1001, "user_alice");
        table->buckets[2] = create_node(1002, "user_bob");
        table->buckets[2]->next = create_node(1003, "user_charlie");
        table->buckets[5] = create_node(1004, "user_diana");
        table->total_items = 4;
        
        printf("Created hash table: %s with %zu items\n", 
               table->table_name, table->total_items);
        
        for (size_t i = 0; i < table->bucket_count; i++) {
            if (table->buckets[i]) {
                printf("  Bucket[%zu]:\n", i);
                Node *node = table->buckets[i];
                while (node) {
                    printf("    %s: %d\n", node->label, node->value);
                    node = node->next;
                }
            }
        }
    }
}

void demonstrate_nested_scope_cleanup() {
    printf("\n=== Nested Scope Cleanup Demo ===\n");
    
    ComplexStruct *outer_struct = create_complex_struct("OuterScope", 5, 3);
    defer(
        printf("Cleaning up outer scope\n");
        free_complex_struct(outer_struct);
    );
    
    {
        ComplexStruct *inner_struct = create_complex_struct("InnerScope", 3, 2);
        defer(
            printf("Cleaning up inner scope\n");
            free_complex_struct(inner_struct);
        );
        
        Node *temp_node = create_node(999, "temporary");
        defer(
            printf("Cleaning up temporary node\n");
            free_node_chain(temp_node);
        );
        
        if (outer_struct && inner_struct && temp_node) {
            printf("All allocations successful in nested scope\n");
            printf("Outer: %s, Inner: %s, Temp: %s\n", 
                   outer_struct->name, inner_struct->name, temp_node->label);
        }
        
        printf("About to exit inner scope...\n");
    }
    
    printf("Back in outer scope\n");
}

int main(void) {
    printf("Starting complex allocation demonstrations...\n");
    
    demonstrate_complex_allocations();
    demonstrate_linked_list();
    demonstrate_hash_table();
    demonstrate_nested_scope_cleanup();
    
    printf("\nAll demonstrations complete.\n");
    return EXIT_SUCCESS;
}
