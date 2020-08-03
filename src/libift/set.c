#include <stdlib.h>
#include "set.h"


void destroySet(Set **head_address)
{
    Set *head = *head_address;
    while (head)
    {
        Set *next = head->next;
        free(head);
        head = next;
    }
    *head_address = NULL;
}


void pushSet(Set **head_address, int value)
{
    Set *node = malloc(sizeof *node);
    if (!node)
    {
        destroySet(head_address);
        *head_address = NULL;
        return;
    }

    node->value = value;
    node->next = *head_address;
    *head_address = node;
}


int lengthSet(const Set *head)
{
   int length = 0;
   for (const Set *s = head; s; s = s->next)
       length++;
   return length;
}


void reverseSet(Set **head_address)
{
    Set *current = *head_address;
    Set *prev = NULL, *next = NULL;
    
    while (current)
    {
        next = current->next;
        current->next = prev;
        prev = current;
        current = next;
    }
    *head_address = prev;
}
