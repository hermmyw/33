long foo();
long bar();
void useless();

long foo()
{
 long a = 0xfeed;
 long b = 0xface;
 long c = bar(a, b) + 1;
 return c;
}

int main()
{
 foo();
}


void useless()
{
 int a = 0;
}

long bar(long a, long b)
{
 unsigned long ret =
 ((unsigned long) (a << 16)) |
 ((unsigned long) b);
 useless();
 return ret;
}

