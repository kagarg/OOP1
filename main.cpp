
#include <iostream>
#include <fstream>
#include <forward_list>
#include <cmath>
#include <random>
#include <stdexcept>
#define interface struct 

interface IDistribution{ // интерфейс распределения
    virtual ~IDistribution() = default;
    virtual double density(double x)const=0; // чисто виртуальные 
    virtual double randNum()const=0; // функции
    virtual double excess()const=0;
    virtual double asymmetry()const=0;
    virtual double mean()const=0;
    virtual double dispersion()const=0;
// . . .
};
interface IPersistent{ // интерфейс персистентного объекта
    virtual void save(std::ofstream& out)const=0;// чисто виртуальные
    virtual void load(std::ifstream& in)=0;// функции
// . . . 
};




class Normal : public IDistribution, public IPersistent{
    double mu_, sigma_; // mu - сдвиг (математическое ожидание), sigma - масштаб (стандартное отклонение)
    mutable std::mt19937 gen_; // mutable т.к. randNum() const
public:
    Normal(double mu, double sigma, uint32_t seed = std::random_device{}())
       : mu_(mu), sigma_(sigma), gen_(seed) {
    if (sigma_ <= 0.0)
        throw std::invalid_argument("sigma must be > 0");
    }; // конструктор

    double density(double x) const override{ // x — значение случайной величины, 𝜇 — математическое ожидание, 𝜎 — стандартное отклонение
        const double pi = std::acos(-1.0);
        return (1.0 / (sigma_ * sqrt(2.0 * pi))) * exp(-0.5 * pow((x - mu_) / sigma_, 2));
    };
    double mean() const override{ return mu_; }
    double dispersion() const override{ return pow(sigma_, 2); }
    double randNum() const override{
        std::normal_distribution<double> dist(mu_, sigma_);
        return dist(gen_);
    };
    double asymmetry() const override{ return 0.0; };
    double excess() const override{ return 0.0; };

    void save(std::ofstream& stream) const override{ stream << mu_ << ' ' << sigma_ << '\n'; };
    void load(std::ifstream& stream) override{
        double mu, sigma;
        stream >> mu >> sigma;
        if (!stream)
            throw std::runtime_error("failed to load Normal");
        if (sigma <= 0.0)
            throw std::invalid_argument("sigma must be > 0");
        mu_ = mu;
        sigma_ = sigma;
    };
    void setMu(double mu){ mu_ = mu; };
    void setSigma(double sigma){
        if (sigma <= 0.0)
        throw std::invalid_argument("sigma must be > 0");
        sigma_ = sigma;
    };

};


class Uniform : public IDistribution, public IPersistent{
    double a_, b_; // a - нижняя граница(сдвиг), b - верхняя граница( сдвиг = b-a )
    mutable std::mt19937 gen_; // mutable т.к. randNum() const
public:
    Uniform(double a, double b, uint32_t seed = std::random_device{}())
        : a_(a), b_(b), gen_(seed){
        if (b_ <= a_)
            throw std::invalid_argument("must be b > a");
    }; // конструктор

    double density(double x) const override{
        if (x < a_ || x > b_) return 0.0;
        return 1.0 / (b_ - a_);
    };
    double mean() const override{
        return (a_ + b_) / 2.0;
    }
    double dispersion() const override{
        return pow(b_ - a_, 2) / 12.0;
    }
    double randNum() const override{
        std::uniform_real_distribution<double> dist(a_, b_);
        return dist(gen_);
    };
    double asymmetry() const override{ return 0.0; }
    double excess() const override{ return -6.0/5.0; }

    void save(std::ofstream& stream) const override{ stream << a_ << ' ' << b_ << '\n'; };
    void load(std::ifstream& stream) override{
        double a, b;
        stream >> a >> b;
        if (!stream)
            throw std::runtime_error("failed to load Uniform");
        if (b <= a)
            throw std::invalid_argument("must be b > a");
        a_ = a;
        b_ = b;
    }
    void setA(double a) {
        if (b_ <= a)
            throw std::invalid_argument("must be b > a");
        a_ = a;
    }

    void setB(double b) {
        if (b <= a_)
            throw std::invalid_argument("must be b > a");
        b_ = b;
    }
};


class ShiftedExponentialLaplace : public IDistribution, public IPersistent{
    double shift_, scale_, form_;
    mutable std::mt19937 gen_; // mutable т.к. randNum() const

    double BaseDensity(double x) const {
        return (form_*(form_+1+ std::abs(x)))/(2.0*pow((form_+std::abs(x)),2.0)) * std::exp(-std::abs(x));
    };

    static double E1(double x) {
        if (x <= 0.0) throw std::invalid_argument("E1(x) is defined only for x > 0");
        return -std::expint(-x);
    }

    double BaseDispersion() const {
        double e1 = E1(form_);
        double var = 1.0 - form_ * std::exp(form_)*e1;
        return 2*form_ * var;
    }
public:
    ShiftedExponentialLaplace(double shift, double scale, double form, uint32_t seed = std::random_device{}())
        : shift_(shift), scale_(scale), form_(form), gen_(seed){
        if (scale <= 0.0) {throw std::invalid_argument("scale must be > 0");}
        if (form <= 0.0) {throw std::invalid_argument("form must be > 0");}
        }; // конструктор   

    double density(double x) const override{ return BaseDensity((x-shift_)/scale_) / scale_; }

    double randNum() const override{
        std::exponential_distribution<double> exp1(1.0);
        std::exponential_distribution<double> expForm(form_);
        std::bernoulli_distribution bern(0.5);

        double e = exp1(gen_);
        double t = expForm(gen_);
        int b = bern(gen_) ? 1 : 0;

        double x0 = e * (2 * b - 1) / (t + 1.0);
        return shift_ + scale_ * x0;
    }
    double mean() const override{
        return shift_;
    }
    double dispersion() const override{
        return scale_ * scale_ * BaseDispersion();
    }
    double asymmetry() const override{
        return 0.0;
    }
    double excess() const override{
        double e1 = E1(form_);
        double var = 1.0 - form_ * std::exp(form_)*e1;
        return (pow(form_,2)* var - form_ + 2.0)/ (form_* pow(var,2)) -3.0;
    }
    void save(std::ofstream& stream) const override {
        stream << shift_ << ' ' << scale_ << ' ' << form_ << '\n';
    }

    void load(std::ifstream& stream) override {
        double shift, scale, form;
        stream >> shift >> scale >> form;
        if (!stream)
            throw std::runtime_error("failed to load ShiftedExponentialLaplace");
        if (scale <= 0.0)
            throw std::invalid_argument("scale must be > 0");
        if (form <= 0.0)
            throw std::invalid_argument("form must be > 0");
        shift_ = shift;
        scale_ = scale;
        form_ = form;
    }

    void setShift(double shift) { shift_ = shift; }

    void setScale(double scale) {
        if (scale <= 0.0)
            throw std::invalid_argument("scale must be > 0");
        scale_ = scale;
    }

    void setForm(double form) {
        if (form <= 0.0)
            throw std::invalid_argument("form must be > 0");
        form_ = form;
    }
};    



interface IObserver{virtual void update()=0;}; 

class Empiric {
// . . .
// список наблюдателей: пара "объект - интересующее событие"
    std::forward_list<std::pair<IObserver*, int>> observers; 
public:
    // . . .
    //добавить наблюдателя
    void attach(IObserver* observer, int interest){
    observers.push_front(std::pair<IObserver*, int>(observer, interest));
    }
    // удалить наблюдателя
        void detach(IObserver* observer, int interest){
    observers.remove(std::pair<IObserver*, int>(observer, interest));
    }
    // уведомить всех наблюдателей, интересующихся событием interest
        void notify(int interest=1){
    for (auto& elem : observers) 
    if(elem.second==interest) elem.first->update();
    }
    // изменить значение i-го наблюдения на значение x
    void changeData(int i, float x){
        //. . . // произвести изменение
        notify(0); // уведомить всех интересующихся
    }
};

//IObserver – замещает функцию update.

class Histogram : public IObserver {
    Empiric &e;
    //. . .
public:
    Histogram(Empiric & em, int k, int interest=0); // k - число столбцов 
                                    // гистограммы
    void update() override{
    // пересчитать гистограмму по измененной выборке
    }
    //. . .
};

int main(){
    // Empiric em(. . .); 
    // // несколько гистограмм разной степени грубости 
    // Histogram h1(em, 5, 1), h2(em, 10), h3(em, 15);
    //         //. . .
    // // последовательное изменение трех элементов
    // em.changeData(0, 1.2); // обновляются h2 и h3
    // em.changeData(1, 2.3); // обновляются h2 и h3
    // em.changeData(2, 3.4); // обновляются h2 и h3
    // em.notify(); // вызов обновления клиентом, обновляется h1
    //         //. . .
    Normal dist(0.0, 1.0);
    std::cout << "Normal distribution:" << std::endl;
    std::cout << "Density at 0: " << dist.density(0.0) << std::endl;
    std::cout << "Random number: " << dist.randNum() << std::endl;

    std::cout << "Uniform distribution:" << std::endl;
    Uniform dist2(0.0, 1.0);
    std::cout << "Density at 0.5: " << dist2.density    (0.5) << std::endl;
    std::cout << "Random number: " << dist2.randNum() << std::endl;

    std::cout << "Shifted Exponential Laplace distribution:" << std::endl;
    ShiftedExponentialLaplace dist3(0.1, 1.0, 2.0);
    std::cout << "Density at 0: " << dist3.density(0.0) << std::endl;
    std::cout << "Random number: " << dist3.randNum() << std::endl;
    std::cout << "Mean: " << dist3.mean() << std::endl;
    std::cout << "Dispersion: " << dist3.dispersion() << std::endl;
    std::cout << "Asymmetry: " << dist3.asymmetry() << std::endl;
    std::cout << "Excess: " << dist3.excess() << std::endl;

    return 0;
}
