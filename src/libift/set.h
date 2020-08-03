
#ifndef IFT_SET_H
#define IFT_SET_H

#ifdef __cplusplus
extern "C" {
#endif


typedef struct ift_set
{
    int value;
    struct ift_set *next;
} Set;


void destroySet(Set **head_address);
void pushSet(Set **head_address, int value);
int lengthSet(const Set *head);
void reverseSet(Set **head_address);


#ifdef __cplusplus
}
#endif

#endif // IFT_SET_H
